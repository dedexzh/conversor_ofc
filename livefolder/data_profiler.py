"""Perfilamento estatístico de coluna — vetorizado (Polars), sem loop por
célula. Só descreve os VALORES e a ESTRUTURA da coluna; não decide tipo
(isso é `type_inference.py`) e não usa o nome da coluna como sinal.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from . import ptbr_parser
from .models import ColumnProfile, Thresholds, ValueClass

DATE_FORMATS = [
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%y",
]

_BOOL_PARES = [
    {"s", "n"},
    {"sim", "nao", "não"},
    {"true", "false"},
    {"verdadeiro", "falso"},
    {"y", "n"},
]

_NUMERIC_SHAPES = {
    ValueClass.INTEGER_PLAIN.value,
    ValueClass.DECIMAL_COMMA.value,
    ValueClass.MULTI_DOT_GROUPED.value,
    ValueClass.SINGLE_DOT_OTHER_DIGITS.value,
    ValueClass.SINGLE_DOT_3DIGITS.value,
}
_DECIMAL_SHAPES = {ValueClass.DECIMAL_COMMA.value, ValueClass.SINGLE_DOT_OTHER_DIGITS.value}


def pct_date_lazy(non_empty: pl.Series) -> float:
    """Igual ao cálculo que `profile_column` fazia sempre — mas chamado sob
    demanda (só quando a coluna chega na checagem de data em `type_inference`),
    não pra toda coluna de um arquivo mesmo quando ela já bateu como número/
    identificador/booleano antes disso."""
    return _pct_date(non_empty)


def _pct_date(non_empty: pl.Series) -> float:
    n = non_empty.len()
    if n == 0:
        return 0.0
    melhor = 0.0
    for fmt in DATE_FORMATS:
        try:
            parsed = non_empty.str.strptime(pl.Datetime, fmt, strict=False)
        except Exception:
            continue
        taxa = float(parsed.is_not_null().sum()) / n
        melhor = max(melhor, taxa)
    return melhor


def _pct_boolean(non_empty: pl.Series) -> float:
    if non_empty.len() == 0:
        return 0.0
    valores = set(non_empty.str.to_lowercase().str.strip_chars().unique().to_list())
    valores.discard("")
    if not valores:
        return 0.0
    for par in _BOOL_PARES:
        if valores <= par:
            return 1.0
    return 0.0


def profile_column(
    series: pl.Series, thresholds: Optional[Thresholds] = None, shapes_full: Optional[pl.Series] = None
) -> ColumnProfile:
    """Perfila uma coluna (Series de string, RAW). Vetorizado de ponta a ponta.

    `shapes_full` (mesmo tamanho/ordem de `series`) é a classe de forma já
    calculada por `ptbr_parser.batch_materialize_core_shape` para TODAS as
    colunas do arquivo de uma vez — evita recomputar `classify_series` aqui
    (chamado por coluna, é onde o custo por-coluna se acumulava). Sem esse
    parâmetro, calcula por conta própria (usado em testes/uso isolado)."""
    thresholds = thresholds or Thresholds()
    name = series.name or ""
    n_total = series.len()
    n_null = int(series.null_count())

    s_str = series.cast(pl.Utf8, strict=False)
    stripped = s_str.str.strip_chars()
    is_blank_mask = (stripped == "").fill_null(False)
    n_blank = int(is_blank_mask.sum())

    non_empty_mask = series.is_not_null() & (~is_blank_mask)
    non_empty = stripped.filter(non_empty_mask)
    n_non_empty = non_empty.len()

    if n_non_empty == 0:
        return ColumnProfile(name=name, n_total=n_total, n_null=n_null, n_blank=n_blank, sample_values=[])

    n_distinct = int(non_empty.n_unique())
    lens = non_empty.str.len_chars()
    min_len = int(lens.min())
    max_len = int(lens.max())
    fixed_width = min_len == max_len

    has_leading_zero = bool(non_empty.str.contains(r"^0\d").any())
    has_alpha = bool(non_empty.str.contains(r"[A-Za-z]").any())
    has_currency_symbol = bool(non_empty.str.contains(r"[R\$€]").any())
    has_percent_symbol = bool(non_empty.str.contains("%").any())
    digits_only_mask = non_empty.str.contains(r"^\d+$")
    digits_only_pct = float(digits_only_mask.sum()) / n_non_empty

    if shapes_full is not None:
        shapes = shapes_full.filter(non_empty_mask)
    else:
        shapes = (
            pl.DataFrame({"v": non_empty})
            .select(ptbr_parser.classify_series(pl.col("v")).alias("shape"))["shape"]
        )

    has_decimal_comma = bool((shapes == ValueClass.DECIMAL_COMMA.value).any())
    has_multi_dot = bool((shapes == ValueClass.MULTI_DOT_GROUPED.value).any())
    has_single_dot_other = bool((shapes == ValueClass.SINGLE_DOT_OTHER_DIGITS.value).any())
    has_single_dot_3 = bool((shapes == ValueClass.SINGLE_DOT_3DIGITS.value).any())

    numeric_mask = shapes.is_in(list(_NUMERIC_SHAPES))
    pct_numeric = float(numeric_mask.sum()) / n_non_empty

    integer_mask = shapes == ValueClass.INTEGER_PLAIN.value
    pct_integer = float(integer_mask.sum()) / n_non_empty

    decimal_mask = shapes.is_in(list(_DECIMAL_SHAPES))
    pct_decimal = float(decimal_mask.sum()) / n_non_empty

    # pct_date NÃO é calculado aqui de propósito: tentar 7 formatos de data
    # em toda coluna, mesmo nas que já vão bater como numérica/identificador/
    # booleana antes de chegar na checagem de data, era desperdício mensurável
    # num arquivo real. type_inference chama data_profiler.pct_date_lazy()
    # sob demanda, só quando a coluna realmente chega na checagem de data.
    pct_date = 0.0
    pct_boolean = _pct_boolean(non_empty)

    return ColumnProfile(
        name=name,
        n_total=n_total,
        n_null=n_null,
        n_blank=n_blank,
        n_distinct=n_distinct,
        min_len=min_len,
        max_len=max_len,
        fixed_width=fixed_width,
        pct_numeric=pct_numeric,
        pct_integer=pct_integer,
        pct_decimal=pct_decimal,
        pct_date=pct_date,
        pct_boolean=pct_boolean,
        has_thousands_sep_dot=has_multi_dot or has_single_dot_3,
        has_decimal_comma=has_decimal_comma,
        has_decimal_dot=has_single_dot_other,
        has_currency_symbol=has_currency_symbol,
        has_percent_symbol=has_percent_symbol,
        has_alpha=has_alpha,
        has_leading_zero=has_leading_zero,
        digits_only_pct=digits_only_pct,
        sample_values=non_empty.head(6).to_list(),
    )
