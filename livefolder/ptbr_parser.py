"""Parser numérico brasileiro (PT-BR/SAP) — núcleo do motor de tipagem.

Resolve a ambiguidade central de exports brasileiros — "6.000" é milhar (BR)
ou decimal (ISO)? — em duas etapas deliberadamente separadas:

  Fase A (classify_value_shape / classify_series): classifica a FORMA de cada
  valor (não decide o número ainda). Um valor como "6.000" é, isoladamente,
  ambíguo — mas "1.234,56" ou "6,5" em QUALQUER outro valor da MESMA coluna
  provam a convenção do arquivo inteiro (é assim que o Power Query resolve
  isso: uma vez por coluna, não por célula).

  Fase B (infer_column_convention): com a forma de todos os valores da
  coluna, decide a convenção (BR/ISO/UNKNOWN/CONFLICT) uma única vez.

  Fase C (parse_value / parse_series): converte cada valor pra uma STRING
  DECIMAL CANÔNICA (nunca float — evita qualquer perda de precisão em campos
  financeiros; o cast final pra NUMERIC/INTEGER acontece no Postgres via
  to_sql(dtype=...)), usando a convenção já resolvida.

Duas implementações lado a lado, deliberadamente:
  - as funções `_valor` (pure Python) são a REFERÊNCIA/especificação: claras,
    testadas exaustivamente contra cada exemplo do pedido original.
  - as funções `_series` (Polars) são o CAMINHO DE PRODUÇÃO: vetorizadas de
    ponta a ponta (nada de .map_elements/loop por célula em cima de milhões
    de linhas). Usam os MESMOS padrões de regex das funções `_valor` (as
    constantes RE_* abaixo), então não podem divergir por descuido.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional, Sequence

import polars as pl

from .models import ColumnConvention, ParsedValue, ParseState, Thresholds, ValueClass

# ==========================================================================
# Padrões (compartilhados entre a implementação pure-Python e a vetorizada —
# sintaxe compatível entre `re` do Python e o motor de regex do Polars/Rust
# para os padrões abaixo, que só usam \d, âncoras, grupos e quantificadores).
# ==========================================================================
RE_INTEGER_PLAIN = r"^\d+$"
# Milhar '.' (opcional, bem formado) + decimal ',' -> cobre "6,5" (zero
# grupos de milhar) e "1.250.000,75" (dois grupos) — OU um run de dígitos
# sem NENHUM separador de milhar, de qualquer comprimento (achado real:
# export de largura fixa zero-padded como " 00000075165,00", 11 dígitos
# antes da vírgula sem pontos — não é "mal agrupado", só não usa separador
# de milhar nenhum. Sem essa segunda alternativa, esse valor caía em
# MALFORMED e virava NULL mesmo sendo um número BR válido).
RE_DECIMAL_COMMA = r"^(?:\d{1,3}(?:\.\d{3})*|\d+),\d+$"
# 2+ grupos de milhar, todos com 3 dígitos, sem vírgula -> inequívoco BR.
RE_MULTI_DOT_GROUPED = r"^\d{1,3}(\.\d{3}){2,}$"
# Exatamente 1 ponto, exatamente 3 dígitos depois -> AMBÍGUO no valor isolado.
RE_SINGLE_DOT_3DIGITS = r"^\d{1,3}\.\d{3}$"
# Exatamente 1 ponto, dígitos depois != 3 -> só pode ser decimal ISO (milhar
# BR sempre tem grupos de exatamente 3 dígitos).
RE_SINGLE_DOT_OTHER = r"^\d+\.\d+$"

_RE_INTEGER_PLAIN = re.compile(RE_INTEGER_PLAIN)
_RE_DECIMAL_COMMA = re.compile(RE_DECIMAL_COMMA)
_RE_MULTI_DOT_GROUPED = re.compile(RE_MULTI_DOT_GROUPED)
_RE_SINGLE_DOT_3DIGITS = re.compile(RE_SINGLE_DOT_3DIGITS)
_RE_SINGLE_DOT_OTHER = re.compile(RE_SINGLE_DOT_OTHER)

# Símbolos removidos antes da classificação (moeda, percentual, espaços).
_RE_SIMBOLOS = re.compile(r"[R\$€%\s]")


# ==========================================================================
# Fase A — classificação de forma (pure Python, referência)
# ==========================================================================
def _strip_sign_and_symbols(raw: str) -> tuple[str, bool]:
    """Remove sinal (negativo à direita 'SAP', entre parênteses, ou à
    esquerda) e símbolos de moeda/percentual/espaço. Retorna (núcleo, negativo)."""
    v = raw.strip()
    negativo = False
    if v.endswith("-"):
        negativo = True
        v = v[:-1].strip()
    if v.startswith("(") and v.endswith(")"):
        negativo = True
        v = v[1:-1].strip()
    if v.startswith("-"):
        negativo = True
        v = v[1:].strip()
    elif v.startswith("+"):
        v = v[1:].strip()
    v = _RE_SIMBOLOS.sub("", v)
    return v, negativo


def classify_value_shape(raw: Optional[str]) -> tuple[ValueClass, str, bool]:
    """Classifica a forma de um valor textual. Retorna (classe, núcleo_sem_sinal, negativo).

    O núcleo retornado é o texto já sem sinal/símbolos, pronto para a Fase C
    montar a string decimal canônica — evita repetir o strip.
    """
    if raw is None:
        return ValueClass.EMPTY, "", False
    v = str(raw).strip()
    if not v or v.lower() in ("nan", "none", "null", "-", "--"):
        return ValueClass.EMPTY, "", False

    core, negativo = _strip_sign_and_symbols(v)
    if not core:
        return ValueClass.EMPTY, "", negativo

    if _RE_INTEGER_PLAIN.match(core):
        return ValueClass.INTEGER_PLAIN, core, negativo

    if "," in core:
        if _RE_DECIMAL_COMMA.match(core):
            return ValueClass.DECIMAL_COMMA, core, negativo
        return ValueClass.MALFORMED, core, negativo

    n_dots = core.count(".")
    if n_dots >= 2:
        if _RE_MULTI_DOT_GROUPED.match(core):
            return ValueClass.MULTI_DOT_GROUPED, core, negativo
        return ValueClass.MALFORMED, core, negativo

    if n_dots == 1:
        if _RE_SINGLE_DOT_3DIGITS.match(core):
            return ValueClass.SINGLE_DOT_3DIGITS, core, negativo
        if _RE_SINGLE_DOT_OTHER.match(core):
            return ValueClass.SINGLE_DOT_OTHER_DIGITS, core, negativo
        return ValueClass.MALFORMED, core, negativo

    return ValueClass.NOT_NUMERIC_SHAPED, core, negativo


# ==========================================================================
# Fase B — resolução da convenção da coluna
# ==========================================================================
_CLASSES_INEQUIVOCAS_BR = {ValueClass.DECIMAL_COMMA, ValueClass.MULTI_DOT_GROUPED}


def infer_column_convention(
    value_classes: Iterable[ValueClass], thresholds: Optional[Thresholds] = None
) -> ColumnConvention:
    """Decide a convenção numérica (BR/ISO/UNKNOWN/CONFLICT) para uma coluna
    inteira, a partir das formas observadas em TODOS os seus valores.

    Uma única vírgula ou um agrupamento de milhar de 2+ grupos em QUALQUER
    valor da coluna já prova BR de forma inequívoca (arquivos não misturam
    convenção dentro da mesma coluna). Ponto único com != 3 dígitos só pode
    ser decimal ISO. Se as duas evidências aparecerem juntas, é conflito real
    — não adivinha.
    """
    thresholds = thresholds or Thresholds()
    contagem = Counter(value_classes)

    tem_br_inequivoco = bool(contagem[ValueClass.DECIMAL_COMMA] or contagem[ValueClass.MULTI_DOT_GROUPED])
    tem_iso_inequivoco = bool(contagem[ValueClass.SINGLE_DOT_OTHER_DIGITS])

    if tem_br_inequivoco and tem_iso_inequivoco:
        return ColumnConvention.CONFLICT
    if tem_br_inequivoco:
        return ColumnConvention.BR
    if tem_iso_inequivoco:
        return ColumnConvention.ISO

    if contagem[ValueClass.SINGLE_DOT_3DIGITS] or contagem[ValueClass.INTEGER_PLAIN]:
        return ColumnConvention.BR if thresholds.resolve_unknown_convention_as_br else ColumnConvention.UNKNOWN

    return ColumnConvention.UNKNOWN


# ==========================================================================
# Fase C — parse final por valor (pure Python, referência)
# ==========================================================================
def _montar_normalizado(classe: ValueClass, core: str, negativo: bool, convention: ColumnConvention) -> Optional[str]:
    """Monta a string decimal canônica (ex.: "6000.50") a partir do núcleo já
    classificado. Retorna None se a classe não permitir montar um número
    (chamador decide AMBIGUOUS/INVALID)."""
    if classe == ValueClass.INTEGER_PLAIN:
        digitos = core.lstrip("0") or "0"
        resultado = digitos
    elif classe in (ValueClass.DECIMAL_COMMA, ValueClass.MULTI_DOT_GROUPED):
        resultado = core.replace(".", "").replace(",", ".")
    elif classe == ValueClass.SINGLE_DOT_OTHER_DIGITS:
        resultado = core
    elif classe == ValueClass.SINGLE_DOT_3DIGITS:
        if convention == ColumnConvention.BR:
            resultado = core.replace(".", "")
        elif convention == ColumnConvention.ISO:
            resultado = core
        else:
            return None
    else:
        return None
    return f"-{resultado}" if negativo and resultado != "0" else resultado


def parse_value(raw: Optional[str], convention: ColumnConvention, thresholds: Optional[Thresholds] = None) -> ParsedValue:
    """Converte um valor textual único, dada a convenção já resolvida da
    coluna a que ele pertence (ver infer_column_convention)."""
    thresholds = thresholds or Thresholds()
    raw_str = "" if raw is None else str(raw)
    classe, core, negativo = classify_value_shape(raw)

    if classe == ValueClass.EMPTY:
        return ParsedValue(raw=raw_str, normalized=None, state=ParseState.EMPTY, value_class=classe)

    if classe in (ValueClass.NOT_NUMERIC_SHAPED, ValueClass.MALFORMED):
        return ParsedValue(raw=raw_str, normalized=None, state=ParseState.INVALID, value_class=classe,
                            notes="não reconhecido como número")

    normalizado = _montar_normalizado(classe, core, negativo, convention)
    if normalizado is None:
        return ParsedValue(raw=raw_str, normalized=None, state=ParseState.AMBIGUOUS, value_class=classe,
                            notes="convenção numérica não resolvida para a coluna (sem vírgula/milhar em nenhum valor)")

    return ParsedValue(raw=raw_str, normalized=normalizado, state=ParseState.NORMALIZED, value_class=classe)


def parse_values(
    raws: Sequence[Optional[str]], thresholds: Optional[Thresholds] = None
) -> tuple[list[ParsedValue], ColumnConvention]:
    """Pipeline completo (Fases A+B+C) em Python puro, para uma coluna inteira
    representada como lista. Referência/uso em testes e arquivos pequenos —
    para DataFrames Polars use `parse_series`."""
    thresholds = thresholds or Thresholds()
    classes = [classify_value_shape(r)[0] for r in raws]
    convention = infer_column_convention(classes, thresholds)
    valores = [parse_value(r, convention, thresholds) for r in raws]
    return valores, convention


# ==========================================================================
# Caminho vetorizado (Polars) — mesma lógica das Fases A/B/C, sem loop Python
# por célula. Usado por data_profiler.py e normalization_engine.py.
# ==========================================================================
def _core_expr(col: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Vetorizado: remove sinal e símbolos, retorna (núcleo, é_negativo)."""
    v = col.cast(pl.Utf8).str.strip_chars()
    termina_menos = v.str.ends_with("-")
    entre_parens = v.str.starts_with("(") & v.str.ends_with(")")
    comeca_menos = v.str.starts_with("-")
    negativo = termina_menos | entre_parens | comeca_menos

    sem_sinal = (
        pl.when(termina_menos).then(v.str.slice(0, v.str.len_chars() - 1))
        .when(entre_parens).then(v.str.slice(1, v.str.len_chars() - 2))
        .when(comeca_menos).then(v.str.slice(1))
        .otherwise(v)
    ).str.strip_chars()
    core = sem_sinal.str.replace_all(r"[R\$€%\s]", "")
    return core, negativo


def classify_series(col: pl.Expr, precomputed_core: Optional[pl.Expr] = None) -> pl.Expr:
    """Vetorizado: Fase A — classifica a forma de cada valor da coluna.

    Aceita `precomputed_core` (de uma chamada anterior a `_core_expr`) pra
    quem já tem o núcleo calculado — evita reavaliar `_core_expr` como
    subexpressão aninhada, que é caro (ver `batch_materialize_core_shape`,
    onde isso importa de verdade: 80 colunas x reavaliar tudo 2-3x cada vira
    minutos em vez de segundos)."""
    core = precomputed_core if precomputed_core is not None else _core_expr(col)[0]
    vazio = col.is_null() | (core == "") | core.str.to_lowercase().is_in(["nan", "none", "null", "-", "--"])
    n_dots = core.str.count_matches(r"\.")
    tem_virgula = core.str.contains(",")

    return (
        pl.when(vazio).then(pl.lit(ValueClass.EMPTY.value))
        .when(core.str.contains(RE_INTEGER_PLAIN)).then(pl.lit(ValueClass.INTEGER_PLAIN.value))
        .when(tem_virgula & core.str.contains(RE_DECIMAL_COMMA)).then(pl.lit(ValueClass.DECIMAL_COMMA.value))
        .when(tem_virgula).then(pl.lit(ValueClass.MALFORMED.value))
        .when((n_dots >= 2) & core.str.contains(RE_MULTI_DOT_GROUPED)).then(pl.lit(ValueClass.MULTI_DOT_GROUPED.value))
        .when(n_dots >= 2).then(pl.lit(ValueClass.MALFORMED.value))
        .when((n_dots == 1) & core.str.contains(RE_SINGLE_DOT_3DIGITS)).then(pl.lit(ValueClass.SINGLE_DOT_3DIGITS.value))
        .when((n_dots == 1) & core.str.contains(RE_SINGLE_DOT_OTHER)).then(pl.lit(ValueClass.SINGLE_DOT_OTHER_DIGITS.value))
        .when(n_dots == 1).then(pl.lit(ValueClass.MALFORMED.value))
        .otherwise(pl.lit(ValueClass.NOT_NUMERIC_SHAPED.value))
    )


def infer_column_convention_from_series(shapes: pl.Series, thresholds: Optional[Thresholds] = None) -> ColumnConvention:
    """Vetorizado: Fase B, usando a coluna de classes já calculada por
    classify_series (agregações .any()/.sum() — não é um loop por célula)."""
    thresholds = thresholds or Thresholds()
    tem_br = bool(
        (shapes == ValueClass.DECIMAL_COMMA.value).any() or (shapes == ValueClass.MULTI_DOT_GROUPED.value).any()
    )
    tem_iso = bool((shapes == ValueClass.SINGLE_DOT_OTHER_DIGITS.value).any())
    if tem_br and tem_iso:
        return ColumnConvention.CONFLICT
    if tem_br:
        return ColumnConvention.BR
    if tem_iso:
        return ColumnConvention.ISO
    tem_candidato = bool(
        (shapes == ValueClass.SINGLE_DOT_3DIGITS.value).any() or (shapes == ValueClass.INTEGER_PLAIN.value).any()
    )
    if tem_candidato:
        return ColumnConvention.BR if thresholds.resolve_unknown_convention_as_br else ColumnConvention.UNKNOWN
    return ColumnConvention.UNKNOWN


def parse_series(
    col: pl.Expr,
    convention: ColumnConvention,
    precomputed_core: Optional[pl.Expr] = None,
    precomputed_negativo: Optional[pl.Expr] = None,
    precomputed_shape: Optional[pl.Expr] = None,
) -> pl.Expr:
    """Vetorizado: Fase C — monta a string decimal canônica pra cada valor,
    dada a convenção já resolvida da coluna (infer_column_convention_from_series).

    Aceita core/negativo/shape pré-computados (ver `batch_materialize_core_shape`)
    pra não reavaliar `_core_expr`/`classify_series` como subexpressão aninhada
    — chamado sem eles (como nos testes, em cima de poucos valores) continua
    correto, só que recomputa tudo internamente."""
    if precomputed_core is not None and precomputed_negativo is not None:
        core, negativo = precomputed_core, precomputed_negativo
    else:
        core, negativo = _core_expr(col)
    shape = precomputed_shape if precomputed_shape is not None else classify_series(col, precomputed_core=core)

    sem_zeros_esq = core.str.strip_chars_start("0")
    inteiro = pl.when(sem_zeros_esq == "").then(pl.lit("0")).otherwise(sem_zeros_esq)
    br_forma = core.str.replace_all(r"\.", "").str.replace(",", ".", literal=True)

    single_dot_resolvido = (
        pl.when(pl.lit(convention == ColumnConvention.BR)).then(core.str.replace_all(r"\.", ""))
        .when(pl.lit(convention == ColumnConvention.ISO)).then(core)
        .otherwise(None)
    )

    resultado_sem_sinal = (
        pl.when(shape == ValueClass.INTEGER_PLAIN.value).then(inteiro)
        .when(shape.is_in([ValueClass.DECIMAL_COMMA.value, ValueClass.MULTI_DOT_GROUPED.value])).then(br_forma)
        .when(shape == ValueClass.SINGLE_DOT_OTHER_DIGITS.value).then(core)
        .when(shape == ValueClass.SINGLE_DOT_3DIGITS.value).then(single_dot_resolvido)
        .otherwise(None)
    )

    return (
        pl.when(resultado_sem_sinal.is_null()).then(None)
        .when(negativo & (resultado_sem_sinal != "0")).then(pl.concat_str([pl.lit("-"), resultado_sem_sinal]))
        .otherwise(resultado_sem_sinal)
    )


# ==========================================================================
# Orquestração em lote — UM select para todas as colunas do arquivo, não um
# select por coluna. Esse é o caminho real usado por `pipeline.py`: chamar
# classify_series/parse_series coluna a coluna (ainda que cada chamada seja
# vetorizada) mostrou custo de ~10-40s por arquivo real de ~80 colunas —
# cada `.select()` do Polars tem overhead fixo por chamada, e uma expressão
# com `classify_series` aninhado dentro de `parse_series` reavaliava a
# classificação inteira de novo. Medido: decompor em "materializar core/
# negativo/forma" (1 select) + "montar normalizado usando essas colunas já
# concretas" (1 select) pro arquivo inteiro cai de ~40s pra ~4s.
# ==========================================================================
def _prefixo_core(nome: str) -> str:
    return f"__core__{nome}"


def _prefixo_neg(nome: str) -> str:
    return f"__neg__{nome}"


def _prefixo_shape(nome: str) -> str:
    return f"__shape__{nome}"


def batch_materialize_core_shape(df: pl.DataFrame) -> pl.DataFrame:
    """Para TODAS as colunas de `df` de uma vez: núcleo (sem sinal/símbolo),
    sinal negativo, e classe de forma — em UM único `.select()`."""
    exprs = []
    for nome in df.columns:
        core_e, neg_e = _core_expr(pl.col(nome))
        exprs.append(core_e.alias(_prefixo_core(nome)))
        exprs.append(neg_e.alias(_prefixo_neg(nome)))
        exprs.append(classify_series(pl.col(nome), precomputed_core=core_e).alias(_prefixo_shape(nome)))
    if not exprs:
        return df.clear()
    return df.select(exprs)


def batch_materialize_normalized(
    intermed: pl.DataFrame, colunas: Sequence[str], conventions: dict
) -> pl.DataFrame:
    """Monta a string decimal canônica para TODAS as colunas de uma vez, num
    único `.select()`, usando as colunas __core__/__neg__/__shape__ já
    materializadas por `batch_materialize_core_shape` (não recomputa nada)."""
    exprs = []
    for nome in colunas:
        conv = conventions.get(nome, ColumnConvention.UNKNOWN)
        e = parse_series(
            pl.col(_prefixo_core(nome)),  # placeholder — não usado quando core/negativo são passados
            conv,
            precomputed_core=pl.col(_prefixo_core(nome)),
            precomputed_negativo=pl.col(_prefixo_neg(nome)),
            precomputed_shape=pl.col(_prefixo_shape(nome)),
        )
        exprs.append(e.alias(nome))
    if not exprs:
        return intermed.clear()
    return intermed.select(exprs)
