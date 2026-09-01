"""Aplica uma `TypeDecision` já tomada a uma coluna RAW, produzindo a coluna
tipada final (Polars) + a lista de valores que não bateram (viraram NULL) —
nunca destrói a coluna inteira por causa de um valor inválido.

Colunas NUMERIC continuam como STRING DECIMAL CANÔNICA aqui (nunca float) —
o cast pra `decimal.Decimal` (preciso, sem arredondamento) acontece só na
ponte com o Postgres, em `postgres_writer.py`.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from . import data_profiler, ptbr_parser
from .models import QualityIssue, TargetType, TypeDecision

_LIMITE_ISSUES_POR_COLUNA = 500

_VOCAB_VERDADEIRO = ["s", "sim", "true", "verdadeiro", "y"]
_VOCAB_FALSO = ["n", "nao", "não", "false", "falso"]

_TIPO_PL_INTEIRO = {
    TargetType.SMALLINT: pl.Int16,
    TargetType.INTEGER: pl.Int32,
    TargetType.BIGINT: pl.Int64,
}


def _preparar(series: pl.Series) -> tuple[pl.Series, pl.Series]:
    """Retorna (stripped, tinha_conteudo) — tinha_conteudo=False para
    nulo/branco original (não é erro, é ausência de dado)."""
    s_str = series.cast(pl.Utf8, strict=False)
    stripped = s_str.str.strip_chars()
    is_blank = (stripped == "").fill_null(False)
    tinha_conteudo = series.is_not_null() & (~is_blank)
    return stripped, tinha_conteudo


def _coletar_issues(nome: str, original: pl.Series, falhou_mask: pl.Series, motivo: str) -> list[QualityIssue]:
    indices = falhou_mask.fill_null(False).arg_true().to_list()
    originais = original.cast(pl.Utf8, strict=False)
    issues = []
    for idx in indices[:_LIMITE_ISSUES_POR_COLUNA]:
        valor = originais[idx]
        issues.append(QualityIssue(row=int(idx), column=nome, raw_value="" if valor is None else str(valor), reason=motivo))
    return issues


def _parse_data(stripped: pl.Series) -> pl.Series:
    n = stripped.len()
    if n == 0:
        return stripped.cast(pl.Datetime, strict=False)
    melhor_taxa = -1
    melhor_resultado = None
    for fmt in data_profiler.DATE_FORMATS:
        try:
            parsed = stripped.str.strptime(pl.Datetime, fmt, strict=False)
        except Exception:
            continue
        taxa = int(parsed.is_not_null().sum())
        if taxa > melhor_taxa:
            melhor_taxa = taxa
            melhor_resultado = parsed
    if melhor_resultado is None:
        return pl.Series([None] * n, dtype=pl.Datetime)
    return melhor_resultado


def normalize_column(
    series: pl.Series, decision: TypeDecision, precomputed_normalized: Optional[pl.Series] = None
) -> tuple[pl.Series, list[QualityIssue]]:
    """`precomputed_normalized` (mesmo tamanho/ordem de `series`): string
    decimal já calculada em lote por `ptbr_parser.batch_materialize_normalized`
    pra TODAS as colunas do arquivo — evita reparsear aqui. Só é usada para
    os alvos INTEGER/BIGINT/SMALLINT/NUMERIC; os demais tipos (boolean, data,
    identificador, texto) sempre computam na hora, são baratos."""
    nome = series.name or ""
    stripped, tinha_conteudo = _preparar(series)

    if decision.target_type == TargetType.BOOLEAN:
        minuscula = stripped.str.to_lowercase()
        expr = (
            pl.when(minuscula.is_in(_VOCAB_VERDADEIRO)).then(True)
            .when(minuscula.is_in(_VOCAB_FALSO)).then(False)
            .otherwise(None)
        )
        resultado = pl.DataFrame({"v": minuscula}).select(expr.alias("v"))["v"]
        falhou = tinha_conteudo & resultado.is_null()
        return resultado, _coletar_issues(nome, series, falhou, "não reconhecido como booleano")

    if decision.target_type in _TIPO_PL_INTEIRO:
        if precomputed_normalized is not None:
            normalizado_str = precomputed_normalized
        else:
            normalizado_str = pl.DataFrame({"v": stripped}).select(
                ptbr_parser.parse_series(pl.col("v"), decision.column_convention).alias("v")
            )["v"]
        resultado = normalizado_str.cast(_TIPO_PL_INTEIRO[decision.target_type], strict=False)
        falhou = tinha_conteudo & resultado.is_null()
        return resultado, _coletar_issues(nome, series, falhou, f"não convertido para {decision.target_type.value}")

    if decision.target_type == TargetType.NUMERIC:
        if precomputed_normalized is not None:
            normalizado_str = precomputed_normalized
        else:
            normalizado_str = pl.DataFrame({"v": stripped}).select(
                ptbr_parser.parse_series(pl.col("v"), decision.column_convention).alias("v")
            )["v"]
        falhou = tinha_conteudo & normalizado_str.is_null()
        issues = _coletar_issues(nome, series, falhou, "não convertido para NUMERIC")

        # Quando o tipo vem FIXO (schema já travado da tabela — ver
        # `tipos_fixos` em pipeline.py), o arquivo atual pode ter um valor
        # com mais dígitos inteiros do que a PRECISÃO/ESCALA fixas
        # comportam (ex.: tabela fixou NUMERIC(4,1), arquivo novo tem
        # "13134.5"). Sem essa checagem aqui, o valor passava reto e só
        # estourava no INSERT do Postgres — derrubando o ARQUIVO INTEIRO por
        # causa de UMA linha (achado real). Vira anomalia (NULL + log) igual
        # a qualquer outro valor que não bate com o tipo decidido.
        if decision.precision:
            max_digitos_inteiros = max(decision.precision - (decision.scale or 0), 0)
            # Zero à esquerda não conta como dígito "de verdade" (export de
            # largura fixa como "00000000000,00" tem 11 caracteres mas vale
            # 0) — sem descartar os zeros à esquerda antes de contar, um
            # valor pequeno zero-padded era rejeitado por "excesso de
            # dígitos" que nem existem de fato (achado real).
            sem_zeros_esquerda = normalizado_str.str.extract(r"^-?(\d+)", 1).str.strip_chars_start("0")
            digitos_inteiros = (
                pl.DataFrame({"p": sem_zeros_esquerda})
                .select(pl.when(pl.col("p") == "").then(1).otherwise(pl.col("p").str.len_chars()).alias("d"))["d"]
                .fill_null(0)
            )
            excede = tinha_conteudo & normalizado_str.is_not_null() & (digitos_inteiros > max_digitos_inteiros)
            if excede.any():
                issues = issues + _coletar_issues(
                    nome, series, excede,
                    f"fora da faixa do tipo fixo NUMERIC({decision.precision},{decision.scale or 0}) da tabela",
                )
                normalizado_str = pl.DataFrame({"v": normalizado_str}).select(
                    pl.when(excede).then(None).otherwise(pl.col("v")).alias("v")
                )["v"]

        return normalizado_str, issues

    if decision.target_type == TargetType.VARCHAR and decision.varchar_len and not decision.is_identifier:
        # Mesma lógica do NUMERIC acima, mas para largura fixa de VARCHAR
        # (ex.: tabela fixou VARCHAR(50), arquivo novo tem um texto de 80
        # caracteres nessa coluna) — sem isso, o INSERT inteiro falhava por
        # "value too long for type character varying(50)".
        excede = tinha_conteudo & (stripped.str.len_chars() > decision.varchar_len)
        issues = _coletar_issues(nome, series, excede, f"maior que a largura fixa VARCHAR({decision.varchar_len}) da tabela")
        resultado = pl.DataFrame({"v": stripped}).select(
            pl.when(excede).then(None).otherwise(pl.col("v")).alias("v")
        )["v"]
        return resultado, issues

    if decision.target_type in (TargetType.DATE, TargetType.TIMESTAMP):
        resultado = _parse_data(stripped)
        falhou = tinha_conteudo & resultado.is_null()
        issues = _coletar_issues(nome, series, falhou, "não reconhecido como data")
        if decision.target_type == TargetType.DATE:
            resultado = resultado.dt.date()
        return resultado, issues

    if decision.is_identifier:
        digitos = stripped.str.replace_all(r"[^0-9]", "")
        expr = pl.when(digitos.str.len_chars() > 0).then(digitos).otherwise(stripped)
        resultado = pl.DataFrame({"v": stripped}).select(expr.alias("v"))["v"]
        return resultado, []

    return stripped, []
