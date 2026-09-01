"""Detecção explícita de delimitador de CSV (`;`, `,`, TAB, `|`).

Usa `csv.reader` (stdlib) por candidato — não `str.split` — para não quebrar
um valor decimal como "100,50" em duas colunas quando o candidato é ",", e
para respeitar campos entre aspas contendo o próprio delimitador. Substitui
o `sep=None, engine='python'` (sniffing opaco) do pandas por uma decisão
explícita e reportável.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass

CANDIDATOS = [";", ",", "\t", "|"]  # ';' primeiro: padrão BR/SAP (',' colide com decimal)


@dataclass
class DelimiterResult:
    delimiter: str
    confidence: float
    n_columns: int
    notes: str


def detect_delimiter(texto: str, max_lines: int = 20) -> DelimiterResult:
    linhas = [l for l in texto.splitlines() if l.strip()][:max_lines]
    if not linhas:
        return DelimiterResult(";", 0.0, 0, "arquivo vazio; usando padrão BR ';'")

    melhor: DelimiterResult | None = None
    for delim in CANDIDATOS:
        contagens = []
        for linha in linhas:
            try:
                campos = next(csv.reader([linha], delimiter=delim, quotechar='"'))
            except (csv.Error, StopIteration):
                continue
            contagens.append(len(campos))

        if not contagens:
            continue
        n_colunas, freq = Counter(contagens).most_common(1)[0]
        if n_colunas <= 1:
            continue

        consistencia = freq / len(contagens)
        candidato = DelimiterResult(
            delim, consistencia, n_colunas, f"{freq}/{len(contagens)} linhas com {n_colunas} campos"
        )
        chave_atual = (consistencia, n_colunas)
        chave_melhor = (melhor.confidence, melhor.n_columns) if melhor else (-1.0, -1)
        if chave_atual > chave_melhor:
            melhor = candidato

    if melhor is None:
        return DelimiterResult(";", 0.0, 1, "nenhum candidato produziu múltiplas colunas consistentes; usando padrão BR ';'")
    return melhor
