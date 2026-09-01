"""Limpeza e deduplicação de nomes de coluna.

Relocaliza (sem mudar a lógica) `FileProcessor.limpar_nome_coluna` e o loop de
deduplicação de `sanitizar_dados` do `conversor_db.py` original — já validados
contra os 106 arquivos reais da pasta monitorada.
"""

from __future__ import annotations

import re
from typing import Sequence

_RE_NAO_PALAVRA = re.compile(r"[^\w\s]")
_RE_ESPACOS = re.compile(r"\s+")


def clean_column_name(nome) -> str:
    """Minúsculo, sem pontuação, espaços viram '_'. Idêntico ao
    `limpar_nome_coluna` original."""
    if nome is None:
        return ""
    texto = str(nome).strip()
    if texto.lower() == "nan":
        return ""
    texto = texto.lower()
    texto = _RE_NAO_PALAVRA.sub("", texto)
    return _RE_ESPACOS.sub("_", texto)


def clean_and_dedupe_columns(nomes: Sequence) -> list[str]:
    """Limpa cada nome, substitui vazios/"unnamed" por `coluna_sem_nome_{i}`,
    e garante nomes únicos com sufixo `_2`, `_3`... Nunca perde uma coluna
    silenciosamente por colisão de nome."""
    novas_cols: list[str] = []
    vistos: dict[str, int] = {}
    for i, col in enumerate(nomes):
        nome_limpo = clean_column_name(col)
        if not nome_limpo or "unnamed" in nome_limpo:
            nome_limpo = f"coluna_sem_nome_{i}"

        if nome_limpo in vistos:
            vistos[nome_limpo] += 1
            nome_limpo = f"{nome_limpo}_{vistos[nome_limpo]}"
        else:
            vistos[nome_limpo] = 1

        novas_cols.append(nome_limpo)
    return novas_cols
