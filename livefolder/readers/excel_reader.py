"""Leitor de Excel via Polars + `calamine` (Rust, através do pacote
`fastexcel`) — sem depender do Excel instalado, sem win32. `calamine` lê
.xlsx/.xlsm/.xls/.xlsb com o mesmo engine, então um único caminho substitui
os três motores (openpyxl/xlrd/pyxlsb) do leitor pandas original.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

EXTENSOES_SUPORTADAS = {".xlsx", ".xlsm", ".xls", ".xlsb"}


@dataclass
class ReadDiagnostics:
    engine: str
    n_rows_read: int
    n_columns_read: int
    sheet_name: str | None = None


def read_excel(caminho: str, sheet_name: str | None = None) -> tuple[pl.DataFrame, ReadDiagnostics]:
    try:
        df = pl.read_excel(caminho, sheet_name=sheet_name, engine="calamine", infer_schema_length=0)
    except Exception as e:
        raise ValueError(f"Falha ao ler Excel ({caminho}): {e}") from e

    diag = ReadDiagnostics(engine="calamine", n_rows_read=df.height, n_columns_read=df.width, sheet_name=sheet_name)
    return df, diag
