"""Veredito de qualidade pós-decisão: quantos valores de uma coluna já
tipada viraram NULL por não bater com o tipo decidido, contra os limiares
configuráveis (0% rigor máximo / <0,1% aceita / 0,1%-1% alerta / >1% bloqueia).

Distinto do limiar usado dentro de `type_inference` (`conversao_minima`, que
decide SE a coluna é daquele tipo) — este aqui decide o que fazer com o
arquivo DEPOIS que o tipo já foi escolhido.
"""

from __future__ import annotations

from enum import Enum

from .models import Thresholds


class Verdict(str, Enum):
    STRICT_OK = "STRICT_OK"
    ACCEPT = "ACCEPT"
    WARN = "WARN"
    BLOCK = "BLOCK"


def evaluate_column(n_total: int, n_bad: int, thresholds: Thresholds | None = None) -> Verdict:
    thresholds = thresholds or Thresholds()
    if n_total <= 0 or n_bad <= 0:
        return Verdict.STRICT_OK
    taxa = n_bad / n_total
    if taxa <= thresholds.accept:
        return Verdict.ACCEPT
    if taxa <= thresholds.warn:
        return Verdict.WARN
    return Verdict.BLOCK


def evaluate_file(vereditos_por_coluna: dict[str, Verdict]) -> Verdict:
    """Veredito do arquivo inteiro = o pior veredito entre as colunas."""
    ordem = {Verdict.STRICT_OK: 0, Verdict.ACCEPT: 1, Verdict.WARN: 2, Verdict.BLOCK: 3}
    if not vereditos_por_coluna:
        return Verdict.STRICT_OK
    return max(vereditos_por_coluna.values(), key=lambda v: ordem[v])


def status_para_relatorio(veredito: Verdict) -> str:
    """Rótulo de status pro relatório de qualidade (formato pedido pelo usuário)."""
    return {
        Verdict.STRICT_OK: "SUCCESS",
        Verdict.ACCEPT: "SUCCESS",
        Verdict.WARN: "SUCCESS_WITH_WARNINGS",
        Verdict.BLOCK: "BLOCKED",
    }[veredito]
