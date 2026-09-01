"""Leitor de CSV: encoding e delimitador decididos explicitamente (não o
sniffing opaco de `sep=None` do pandas), leitura em Polars (engine C, rápido),
tudo como string — a tipagem acontece depois, em `type_inference`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import polars as pl

from .. import delimiter_detector, encoding_detector


@dataclass
class ReadDiagnostics:
    encoding: str
    encoding_confidence: float
    encoding_notes: str
    delimiter: str
    delimiter_confidence: float
    n_rows_read: int
    n_columns_read: int


def read_csv(caminho: str) -> tuple[pl.DataFrame, ReadDiagnostics]:
    with open(caminho, "rb") as f:
        raw_bytes = f.read()

    enc = encoding_detector.detect_encoding(raw_bytes)
    texto = raw_bytes.decode(enc.encoding, errors="replace")

    delim = delimiter_detector.detect_delimiter(texto)

    buffer = io.BytesIO(texto.encode("utf-8"))
    try:
        df = pl.read_csv(
            buffer,
            separator=delim.delimiter,
            infer_schema=False,
            truncate_ragged_lines=True,
            quote_char='"',
            encoding="utf8",
        )
    except Exception as e:
        raise ValueError(f"Falha ao ler CSV ({caminho}): {e}") from e

    diag = ReadDiagnostics(
        encoding=enc.encoding,
        encoding_confidence=enc.confidence,
        encoding_notes=enc.notes,
        delimiter=delim.delimiter,
        delimiter_confidence=delim.confidence,
        n_rows_read=df.height,
        n_columns_read=df.width,
    )
    return df, diag
