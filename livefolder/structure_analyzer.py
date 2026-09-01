"""Limpeza estrutural: colunas vazias, linhas de lixo/resumo, linhas vazias.

Relocaliza a lógica de `FileProcessor.sanitizar_dados` (passos 2-4) do
`conversor_db.py` original para Polars, com um relatório estruturado no lugar
da tupla `(df, int, int)` original — mesmas regras, mais visibilidade.

Assume que `df` já passou por `header_cleaner` (colunas limpas/únicas) e que
todas as colunas vieram como string (leitura RAW com dtype string) — é assim
que os readers deste pacote sempre entregam o DataFrame antes da tipagem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

_LIXO_REGEX = r"(?i)^(TOTAL GERAL|TOTAL|SUBTOTAL|RESUMO)"


@dataclass
class StructureReport:
    n_rows_raw: int = 0
    n_cols_raw: int = 0
    n_rows_final: int = 0
    n_cols_final: int = 0
    empty_columns: list[str] = field(default_factory=list)
    junk_rows_dropped: int = 0
    fully_empty_rows_dropped: int = 0


def _is_blank_expr(col: str) -> pl.Expr:
    return pl.col(col).is_null() | (pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars() == "")


def analyze_and_clean(df: pl.DataFrame) -> tuple[pl.DataFrame, StructureReport]:
    report = StructureReport(n_rows_raw=df.height, n_cols_raw=df.width)

    # 1. Colunas 100% vazias (nulas ou só espaço em todas as linhas).
    empty_columns = [c for c in df.columns if df.select(_is_blank_expr(c).all()).item()]
    if empty_columns:
        df = df.drop(empty_columns)
    report.empty_columns = empty_columns

    if df.height == 0 or df.width == 0:
        report.n_rows_final = df.height
        report.n_cols_final = df.width
        return df, report

    # 2. Linhas de "lixo/resumo" — mesma regra original: avalia só as 3
    # primeiras colunas, por posição.
    colunas_a_checar = df.columns[:3]
    mask_lixo = pl.Series([False] * df.height)
    for c in colunas_a_checar:
        col_mask = (
            df[c]
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars()
            .str.contains(_LIXO_REGEX)
        )
        mask_lixo = mask_lixo | col_mask
    report.junk_rows_dropped = int(mask_lixo.sum())
    df = df.filter(~mask_lixo)

    # 3. Linhas totalmente vazias (todas as colunas nulas/brancas).
    linhas_antes = df.height
    if df.width > 0:
        todas_vazias = pl.all_horizontal([_is_blank_expr(c) for c in df.columns])
        df = df.filter(~todas_vazias)
    report.fully_empty_rows_dropped = linhas_antes - df.height

    report.n_rows_final = df.height
    report.n_cols_final = df.width
    return df, report
