"""Árvore de decisão de tipo — o substituto de `TypeProfiler.inferir_tipo_coluna`.

Guiada por VALORES e ESTRUTURA (`ColumnProfile` + as classes de forma do
`ptbr_parser`), não pelo nome da coluna. O nome só é consultado em dois
lugares bem documentados (desempate BOOLEAN-vs-SMALLINT, e um desempate de
último recurso) — nunca como sinal primário.

CPF/CNPJ/EAN são reconhecidos por FORMATO estrutural (comprimento em dígitos
+ dígito verificador) — não por nome — e tipados como TEXT/VARCHAR (decisão
do usuário: mais seguro que preservar como NUMERIC, mesmo que o catálogo
antigo do Power Query tratasse CNPJ como número).
"""

from __future__ import annotations

import re
from typing import Optional

import polars as pl

from . import data_profiler, ptbr_parser
from .models import ColumnConvention, ColumnProfile, ORDEM_INTEIROS, TargetType, Thresholds, TypeDecision

_MAX_AMOSTRA_CHECKSUM = 5000
_LIMIAR_COMPRIMENTO_CONSISTENTE = 0.98

# CPF/CNPJ têm DOIS dígitos verificadores -> um número aleatório de 11/14
# dígitos passa por puro acaso em ~1/11 x 1/11 ≈ 0,8% dos casos. Um cadastro
# real de clientes pode ter uma fração grande de CNPJ inválido/placeholder
# (conta de pessoa física sem CNPJ preenchida com zeros, erro de digitação,
# migração de sistema antigo) sem deixar de ser, estruturalmente, uma coluna
# de CNPJ — então o limiar fica bem acima do acaso (10%, ~12x) mas longe dos
# 90% que exigiriam dado praticamente limpo. EAN tem só UM dígito
# verificador (~10% de acerto por acaso), então precisa de uma margem maior
# pra não confundir uma coluna numérica qualquer com EAN.
_LIMIAR_CHECKSUM_CPF_CNPJ = 0.10
_LIMIAR_CHECKSUM_EAN = 0.50


# ==========================================================================
# Checksums de identificadores brasileiros — estrutura, não nome.
# ==========================================================================
def _cpf_valido(digitos: str) -> bool:
    if len(digitos) != 11 or digitos == digitos[0] * 11:
        return False
    n = [int(c) for c in digitos]
    soma = sum(n[i] * (10 - i) for i in range(9))
    resto = (soma * 10) % 11
    dv1 = 0 if resto >= 10 else resto
    if dv1 != n[9]:
        return False
    soma = sum(n[i] * (11 - i) for i in range(10))
    resto = (soma * 10) % 11
    dv2 = 0 if resto >= 10 else resto
    return dv2 == n[10]


def _cnpj_valido(digitos: str) -> bool:
    if len(digitos) != 14 or digitos == digitos[0] * 14:
        return False
    n = [int(c) for c in digitos]
    pesos1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    soma = sum(n[i] * pesos1[i] for i in range(12))
    resto = soma % 11
    dv1 = 0 if resto < 2 else 11 - resto
    if dv1 != n[12]:
        return False
    pesos2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    soma = sum(n[i] * pesos2[i] for i in range(13))
    resto = soma % 11
    dv2 = 0 if resto < 2 else 11 - resto
    return dv2 == n[13]


def _ean_valido(digitos: str) -> bool:
    if len(digitos) not in (8, 12, 13, 14):
        return False
    n = [int(c) for c in digitos]
    corpo, dv = n[:-1], n[-1]
    soma = 0
    for i, d in enumerate(reversed(corpo)):
        soma += d * (3 if i % 2 == 0 else 1)
    calculado = (10 - (soma % 10)) % 10
    return calculado == dv


def detect_identifier_format(raw_non_empty: pl.Series) -> tuple[Optional[str], float]:
    """CPF (11 díg.) / CNPJ (14 díg.) / EAN (8,12,13,14 díg.), por comprimento
    ~100% consistente + dígito verificador válido numa amostra acima do
    limiar (ver comentário nos limiares acima — bem acima do que puro acaso
    produziria, mas tolerante a cadastros reais com boa parte de dado sujo/
    placeholder). A verificação de checksum é feita em Python (não
    vetorizável de forma simples), mas só numa amostra limitada — nunca sobre
    o arquivo inteiro. Retorna (formato, taxa_valida_observada) — a taxa vira
    a confiança da decisão; (None, 0.0) se nada bateu."""
    apenas_digitos = raw_non_empty.str.replace_all(r"[^0-9]", "")
    n = apenas_digitos.len()
    if n == 0:
        return None, 0.0

    for comprimento, nome, validador in ((11, "CPF", _cpf_valido), (14, "CNPJ", _cnpj_valido)):
        no_comprimento = apenas_digitos.filter(apenas_digitos.str.len_chars() == comprimento)
        if no_comprimento.len() / n < _LIMIAR_COMPRIMENTO_CONSISTENTE:
            continue
        amostra = no_comprimento.head(_MAX_AMOSTRA_CHECKSUM).to_list()
        taxa_valida = sum(1 for v in amostra if validador(v)) / len(amostra)
        if taxa_valida >= _LIMIAR_CHECKSUM_CPF_CNPJ:
            return nome, taxa_valida

    for comprimento in (8, 12, 13, 14):
        no_comprimento = apenas_digitos.filter(apenas_digitos.str.len_chars() == comprimento)
        if no_comprimento.len() / n < _LIMIAR_COMPRIMENTO_CONSISTENTE:
            continue
        amostra = no_comprimento.head(_MAX_AMOSTRA_CHECKSUM).to_list()
        taxa_valida = sum(1 for v in amostra if _ean_valido(v)) / len(amostra)
        if taxa_valida >= _LIMIAR_CHECKSUM_EAN:
            return "EAN", taxa_valida

    return None, 0.0


# ==========================================================================
# Booleano — desempate por nome só no caso 0/1 (documentado, único lugar).
# ==========================================================================
_PREFIXOS_BOOLEANOS = ("ativo", "flag", "possui", "indicador", "is_", "tem_")


def _parece_zero_um(profile: ColumnProfile) -> bool:
    valores = {v.strip() for v in profile.sample_values}
    return bool(valores) and valores <= {"0", "1"}


def _nome_sugere_booleano(name_hint: Optional[str]) -> bool:
    if not name_hint:
        return False
    nome = name_hint.lower()
    return any(p in nome for p in _PREFIXOS_BOOLEANOS)


# ==========================================================================
# A pergunta difícil: quantidade limpa vs. identificador numérico.
# ==========================================================================
_LIMIAR_CARDINALIDADE_IDENTIFICADOR = 0.5

# Prefixos de coluna que, nas exportações SAP/Promax deste projeto, são
# sempre código/identificador (nunca quantidade) — mesmo era usada pelo
# motor antigo (REGEX_PREFIXOS_IDENTIFICADORES) para "cod./código",
# "UNB" (unidade de negócio) e afins. Só entra em jogo como desempate de
# ÚLTIMO RECURSO em looks_like_identifier_despite_clean_numeric (ver
# docstring do módulo) — nunca como sinal primário, e só quando o zero à
# esquerda é real (profile.has_leading_zero), i.e. só quando ignorar o
# nome faria a formatação original (ex.: "UNB Promax"='0817260') se
# perder ao virar INTEGER.
_RE_PREFIXO_CODIGO = re.compile(r'(^|_)(cod(igo)?|unb|pag|oper)($|_)', re.IGNORECASE)


def _nome_sugere_codigo(name_hint: Optional[str]) -> bool:
    return bool(name_hint) and bool(_RE_PREFIXO_CODIGO.search(name_hint))


def _largura_sem_sinal(raw_non_empty: pl.Series) -> tuple[bool, bool]:
    """Largura (fixa?) e zero-à-esquerda, calculados sobre o NÚCLEO sem sinal
    (ver ptbr_parser._core_expr) — um valor negativo como "-0000034" não pode
    parecer "largura variável" só por causa do sinal, na frente de "0000055"
    (mesmos 7 dígitos de magnitude)."""
    core = (
        pl.DataFrame({"v": raw_non_empty})
        .select(ptbr_parser._core_expr(pl.col("v"))[0].alias("core"))["core"]
    )
    larguras = core.str.len_chars()
    fixed_width = int(larguras.min()) == int(larguras.max())
    has_leading_zero = bool(core.str.contains(r'^0\d').any())
    return fixed_width, has_leading_zero


def looks_like_identifier_despite_clean_numeric(
    profile: ColumnProfile, raw_non_empty: pl.Series, name_hint: Optional[str] = None
) -> Optional[str]:
    """Só é chamada quando a coluna já é 100% inteira (CPF/CNPJ/EAN já foram
    pegos antes, independente de "limpeza"). Largura FIXA + zero à esquerda
    (ex.: "Inicial"='0000112' em todas as linhas) é formatação de exibição
    do SAP — seguro converter pra INTEGER/BIGINT. Largura VARIÁVEL + zero à
    esquerda + alta cardinalidade é o perfil de um identificador não
    catalogado (não é CPF/CNPJ/EAN, que já foram pegos antes) — mais seguro
    preservar como TEXT.

    A largura é medida sobre o núcleo SEM SINAL (ver _largura_sem_sinal) —
    senão uma coluna de quantidade genuína que aceita negativo (ex.:
    "Disp."='-0000034' ao lado de '0000055') pareceria "largura variável"
    só pelo sinal e virava TEXT por engano (achado real, ver testes).

    Quando a largura fixa + zero à esquerda não desambiguam (ex.: "Cod."=
    '000347' ou "UNB Promax"='0817260', ambos fixos e de alta cardinalidade,
    indistinguíveis por VALOR de uma quantidade fixa como "Inicial"), o nome
    da coluna entra como desempate de ÚLTIMO RECURSO (ver _nome_sugere_codigo)
    — só quando o zero à esquerda é real, nunca como sinal primário.

    Retorna o MOTIVO da decisão (para auditoria em TypeDecision.reason), ou
    None se a coluna não parece um identificador."""
    if profile.n_non_empty == 0:
        return None
    cardinalidade = profile.n_distinct / profile.n_non_empty
    fixed_width, has_leading_zero = _largura_sem_sinal(raw_non_empty)
    if not has_leading_zero:
        return None
    if (not fixed_width) and cardinalidade > _LIMIAR_CARDINALIDADE_IDENTIFICADOR:
        return "largura variável (sinal à parte) + zero à esquerda + alta cardinalidade: parece identificador, não quantidade"
    if _nome_sugere_codigo(name_hint):
        return "largura fixa + zero à esquerda + nome sugere código/identificador (desempate por nome, último recurso)"
    return None


# ==========================================================================
# Ramo numérico: escolhe SMALLINT/INTEGER/BIGINT/NUMERIC(p,s), vetorizado.
# ==========================================================================
def _decidir_tipo_numerico(normalized: pl.Series) -> TypeDecision:
    validos = normalized.drop_nulls()
    n = validos.len()
    if n == 0:
        return TypeDecision(target_type=TargetType.TEXT, confidence=0.0, reason="sem valores numéricos válidos")

    tem_fracao = bool(validos.str.contains(r"\.").any())

    if not tem_fracao:
        como_int = validos.cast(pl.Int64, strict=False)
        overflow = int(como_int.is_null().sum())
        if overflow > 0:
            precisao = int(validos.str.replace("-", "", literal=True).str.len_chars().max())
            return TypeDecision(target_type=TargetType.NUMERIC, precision=precisao, scale=0, confidence=0.8,
                                 reason="inteiro maior que o limite de BIGINT")
        maximo = como_int.max()
        minimo = como_int.min()
        if -32768 <= minimo and maximo <= 32767:
            alvo = TargetType.SMALLINT
        elif -2147483648 <= minimo and maximo <= 2147483647:
            alvo = TargetType.INTEGER
        else:
            alvo = TargetType.BIGINT
        return TypeDecision(target_type=alvo, confidence=0.9, reason="valores inteiros")

    # A ESCALA final (aplicada a TODA a coluna) é o MAIOR nº de casas decimais
    # vistas em qualquer linha — então a PRECISÃO tem que caber os dígitos
    # INTEIROS da linha mais larga MAIS essa escala, não o comprimento total
    # (inteiro+decimal) de cada linha isolada. Ex.: "5844.9977" (escala 4) e
    # "13134.532" (escala 3, 5 dígitos inteiros) têm o MESMO comprimento total
    # (8), mas a coluna toda escalada em 4 casas decimais precisa de 5+4=9
    # dígitos — não 8 (achado real: numeric field overflow ao gravar
    # "13134.532" como NUMERIC(8,4), que só cabe 4 dígitos inteiros).
    escala_por_valor = validos.str.extract(r"\.(\d+)$", 1).str.len_chars().fill_null(0)
    max_escala = int(escala_por_valor.max() or 0)
    digitos_totais_por_valor = validos.str.replace_all(r"[.\-]", "").str.len_chars()
    max_digitos_inteiros = int((digitos_totais_por_valor - escala_por_valor).max() or 0)
    precisao = max_digitos_inteiros + max_escala
    return TypeDecision(target_type=TargetType.NUMERIC, precision=precisao, scale=max_escala, confidence=0.9,
                         reason="valores decimais")


def _todas_meia_noite(raw_non_empty: pl.Series) -> bool:
    for fmt in data_profiler.DATE_FORMATS:
        try:
            parsed = raw_non_empty.str.strptime(pl.Datetime, fmt, strict=False)
        except Exception:
            continue
        validos = parsed.drop_nulls()
        if validos.len() == 0 or validos.len() / raw_non_empty.len() < 0.5:
            continue
        return bool(((validos.dt.hour() == 0) & (validos.dt.minute() == 0) & (validos.dt.second() == 0)).all())
    return True  # sem confiança suficiente para separar hora -> assume DATE (mais conservador)


# ==========================================================================
# Árvore de decisão principal.
# ==========================================================================
def infer_column_type(
    profile: ColumnProfile,
    raw_non_empty: pl.Series,
    name_hint: Optional[str] = None,
    thresholds: Optional[Thresholds] = None,
    shapes_non_empty: Optional[pl.Series] = None,
    normalized_non_empty: Optional[pl.Series] = None,
) -> TypeDecision:
    """`shapes_non_empty`/`normalized_non_empty`: já calculados em lote pra
    TODAS as colunas do arquivo por `ptbr_parser.batch_materialize_*` — evita
    recomputar `classify_series`/`parse_series` aqui (ver comentário em
    `ptbr_parser.py` sobre o custo de fazer isso coluna a coluna). Sem eles,
    calcula por conta própria (uso isolado/testes)."""
    thresholds = thresholds or Thresholds()

    if profile.n_non_empty == 0:
        return TypeDecision(target_type=TargetType.TEXT, confidence=0.0, reason="coluna vazia")

    # CPF/CNPJ/EAN nunca têm letra — pular a checagem (regex + laço de
    # checksum) pra qualquer coluna que tenha alguma letra é um filtro
    # barato (o profile já calculou has_alpha) que evita rodar isso em
    # colunas de texto puro (nome, endereço, descrição — a maioria de um
    # arquivo real), que não têm a menor chance de ser um desses formatos.
    identifier_kind, taxa_checksum = (None, 0.0) if profile.has_alpha else detect_identifier_format(raw_non_empty)
    if identifier_kind:
        comprimentos_fixos = {"CPF": 11, "CNPJ": 14}
        comprimento = comprimentos_fixos.get(identifier_kind, profile.max_len)
        return TypeDecision(
            target_type=TargetType.VARCHAR, varchar_len=comprimento, is_identifier=True,
            identifier_kind=identifier_kind, confidence=taxa_checksum,
            reason=(
                f"formato {identifier_kind} — {taxa_checksum:.0%} dos valores de {comprimento} dígitos "
                f"passam no dígito verificador (bem acima do que puro acaso produziria)"
            ),
        )

    if profile.pct_boolean >= 0.999:
        # pct_boolean já garante (em data_profiler._pct_boolean) que o
        # conjunto de valores, sem diferenciar maiúsculas/minúsculas, é um
        # subconjunto de um vocabulário booleano conhecido — não precisa
        # (e não deve) exigir n_distinct<=2 aqui, que é sensível a
        # maiúsculas/minúsculas e rejeitaria "SIM"/"sim" misturados à toa.
        return TypeDecision(target_type=TargetType.BOOLEAN, confidence=0.9,
                             reason="conjunto de valores é vocabulário booleano conhecido (S/N, SIM/NAO, ...)")
    if profile.n_distinct == 2 and _parece_zero_um(profile) and _nome_sugere_booleano(name_hint):
        return TypeDecision(target_type=TargetType.BOOLEAN, confidence=0.6,
                             reason="valores 0/1 + nome da coluna sugere flag booleana (desempate por nome)")

    if profile.pct_numeric >= thresholds.conversao_minima:
        if shapes_non_empty is not None:
            shapes = shapes_non_empty
        else:
            shapes = (
                pl.DataFrame({"v": raw_non_empty})
                .select(ptbr_parser.classify_series(pl.col("v")).alias("s"))["s"]
            )
        convention = ptbr_parser.infer_column_convention_from_series(shapes, thresholds)

        if convention == ColumnConvention.CONFLICT:
            return TypeDecision(target_type=TargetType.TEXT, confidence=0.3, column_convention=convention,
                                 reason="convenção numérica conflitante (evidência BR e ISO na mesma coluna)")
        if convention == ColumnConvention.UNKNOWN:
            return TypeDecision(target_type=TargetType.TEXT, confidence=0.3, column_convention=convention,
                                 reason="convenção numérica ambígua, sem resolução automática configurada")

        if normalized_non_empty is not None:
            normalized = normalized_non_empty
        else:
            normalized = (
                pl.DataFrame({"v": raw_non_empty})
                .select(ptbr_parser.parse_series(pl.col("v"), convention).alias("n"))["n"]
            )
        decisao = _decidir_tipo_numerico(normalized)
        decisao.column_convention = convention

        if decisao.target_type in ORDEM_INTEIROS:
            motivo_identificador = looks_like_identifier_despite_clean_numeric(profile, raw_non_empty, name_hint)
            if motivo_identificador:
                return TypeDecision(
                    target_type=TargetType.TEXT, is_identifier=True, confidence=0.6, column_convention=convention,
                    reason=motivo_identificador,
                )
        return decisao

    pct_date = data_profiler.pct_date_lazy(raw_non_empty)
    if pct_date >= thresholds.conversao_minima:
        alvo = TargetType.DATE if _todas_meia_noite(raw_non_empty) else TargetType.TIMESTAMP
        return TypeDecision(target_type=alvo, confidence=0.85, reason="valores batem padrão de data reconhecido")

    if profile.max_len <= 255:
        return TypeDecision(target_type=TargetType.VARCHAR, varchar_len=255, confidence=0.5, reason="texto curto")
    return TypeDecision(target_type=TargetType.TEXT, confidence=0.5, reason="texto longo")


# ==========================================================================
# Conciliação entre arquivos (substitui TypeProfiler.combinar_tipos).
# ==========================================================================
# Dígitos decimais necessários para caber o MAIOR valor absoluto que cada tipo
# inteiro do Postgres permite (ex.: INTEGER vai até 2147483647 -> 10 dígitos).
# Usado só para dimensionar a PRECISÃO de um NUMERIC conciliado com um tipo
# inteiro (ver combinar_tipos) — nunca para validar/converter valor nenhum.
_DIGITOS_MAXIMOS_INTEIRO = {
    TargetType.SMALLINT: 5,   # -32768..32767
    TargetType.INTEGER: 10,   # -2147483648..2147483647
    TargetType.BIGINT: 19,    # -9223372036854775808..9223372036854775807
}


def combinar_tipos(anterior: Optional[TypeDecision], nova: TypeDecision) -> TypeDecision:
    """Concilia o tipo já padronizado globalmente para a coluna (visto em
    outros arquivos) com o tipo inferido no arquivo atual. Regra: só anda em
    direção ao tipo mais amplo/seguro; TEXT nunca regride pra numérico."""
    if anterior is None:
        return nova

    if anterior.target_type == nova.target_type:
        if nova.target_type == TargetType.NUMERIC:
            # Maxar precisão e escala de forma INDEPENDENTE (versão antiga)
            # pode reduzir os dígitos inteiros de um dos lados: NUMERIC(8,2)
            # (6 dígitos inteiros) conciliado com NUMERIC(6,4) (2 dígitos
            # inteiros) dava NUMERIC(8,4) — só 4 dígitos inteiros, menor que
            # os 6 que o primeiro lado já usa. O ALTER COLUMN subsequente
            # estoura para qualquer valor de 5-6 dígitos já gravado (mesma
            # causa raiz do overflow conciliado com inteiro, ver acima) — por
            # isso preserva o maior nº de dígitos INTEIROS dos dois lados.
            escala = max(anterior.scale or 0, nova.scale or 0)
            digitos_inteiros = max(
                (anterior.precision or 0) - (anterior.scale or 0),
                (nova.precision or 0) - (nova.scale or 0),
                0,
            )
            return TypeDecision(
                target_type=TargetType.NUMERIC,
                scale=escala,
                precision=digitos_inteiros + escala,
                confidence=max(anterior.confidence, nova.confidence),
                reason="conciliado (mesmo tipo, dígitos inteiros e escala máximos preservados)",
            )
        if nova.target_type == TargetType.VARCHAR:
            return TypeDecision(
                target_type=TargetType.VARCHAR,
                varchar_len=max(anterior.varchar_len or 0, nova.varchar_len or 0),
                is_identifier=anterior.is_identifier or nova.is_identifier,
                identifier_kind=anterior.identifier_kind or nova.identifier_kind,
                confidence=max(anterior.confidence, nova.confidence),
                reason="conciliado (mesmo tipo, largura máxima)",
            )
        return nova

    tipos = {anterior.target_type, nova.target_type}

    if TargetType.TEXT in tipos:
        return TypeDecision(target_type=TargetType.TEXT, confidence=max(anterior.confidence, nova.confidence),
                             reason="TEXT não regride para outro tipo (uma vez texto, sempre texto)")

    if anterior.target_type in ORDEM_INTEIROS and nova.target_type in ORDEM_INTEIROS:
        vencedor = max(anterior.target_type, nova.target_type, key=lambda t: ORDEM_INTEIROS[t])
        return TypeDecision(target_type=vencedor, confidence=max(anterior.confidence, nova.confidence),
                             reason=f"conciliado entre tipos inteiros (vence o mais amplo: {vencedor.value})")

    if TargetType.NUMERIC in tipos and (ORDEM_INTEIROS.keys() & tipos):
        numerico = anterior if anterior.target_type == TargetType.NUMERIC else nova
        inteiro = nova if anterior.target_type == TargetType.NUMERIC else anterior
        escala = numerico.scale or 0
        # O lado NUMERIC só viu a amostra do PRÓPRIO arquivo que o gerou — sua
        # precisão pode não caber a faixa de valores que o lado INTEIRO já tem
        # (de OUTRO arquivo, já gravado na tabela). Ex.: arquivo A tinha
        # "vl_estouro_limite" só com inteiros até 4 dígitos (vira INTEGER);
        # arquivo B tem casas decimais mas só amostrou valores pequenos (vira
        # NUMERIC(4,1)). Conciliar direto por NUMERIC(4,1) faz o ALTER COLUMN
        # subsequente estourar ao converter os inteiros de 4 dígitos já
        # gravados (achado real: numeric field overflow). Por isso a precisão
        # final garante espaço para a faixa MÁXIMA que o tipo inteiro permite.
        precisao_min_para_inteiro = _DIGITOS_MAXIMOS_INTEIRO.get(inteiro.target_type, 19) + escala
        precisao = max(numerico.precision or 0, precisao_min_para_inteiro)
        return TypeDecision(target_type=TargetType.NUMERIC, scale=escala, precision=precisao,
                             confidence=max(anterior.confidence, nova.confidence),
                             reason="inteiro conciliado com NUMERIC (NUMERIC é mais amplo; precisão ampliada para caber toda a faixa do tipo inteiro)")

    if tipos == {TargetType.DATE, TargetType.TIMESTAMP}:
        return TypeDecision(target_type=TargetType.TIMESTAMP, confidence=max(anterior.confidence, nova.confidence),
                             reason="DATE conciliado com TIMESTAMP (TIMESTAMP é mais amplo)")

    if tipos == {TargetType.VARCHAR, TargetType.TEXT}:
        return TypeDecision(target_type=TargetType.TEXT, confidence=max(anterior.confidence, nova.confidence),
                             reason="VARCHAR conciliado com TEXT")

    return TypeDecision(target_type=TargetType.TEXT, confidence=0.3,
                         reason=f"combinação incompatível ({anterior.target_type.value} x {nova.target_type.value}) -> TEXT")
