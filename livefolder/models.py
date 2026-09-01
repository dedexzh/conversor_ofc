"""Contrato de dados compartilhado entre os módulos do LiveFolder.

Dataclasses/enums puros em Python (sem depender de pandas nem Polars), para
que a lógica de decisão (ptbr_parser, type_inference) seja agnóstica da
biblioteca usada na leitura/perfilamento. Só `readers/*.py`, `data_profiler.py`
e `normalization_engine.py` tocam DataFrames de verdade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ParseState(str, Enum):
    """Estado de uma conversão RAW -> NORMALIZED -> TYPED, por valor."""

    NORMALIZED = "NORMALIZED"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"
    EMPTY = "EMPTY"


class ValueClass(str, Enum):
    """Forma de um valor numérico textual, antes de decidir seu significado."""

    INTEGER_PLAIN = "INTEGER_PLAIN"
    DECIMAL_COMMA = "DECIMAL_COMMA"
    MULTI_DOT_GROUPED = "MULTI_DOT_GROUPED"
    SINGLE_DOT_OTHER_DIGITS = "SINGLE_DOT_OTHER_DIGITS"
    SINGLE_DOT_3DIGITS = "SINGLE_DOT_3DIGITS"
    NOT_NUMERIC_SHAPED = "NOT_NUMERIC_SHAPED"
    MALFORMED = "MALFORMED"
    EMPTY = "EMPTY"


class ColumnConvention(str, Enum):
    """Convenção numérica resolvida para uma coluna inteira."""

    BR = "BR"
    ISO = "ISO"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


class TargetType(str, Enum):
    """Tipo lógico final, mapeado 1:1 para um tipo de coluna do Postgres."""

    BOOLEAN = "BOOLEAN"
    SMALLINT = "SMALLINT"
    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    NUMERIC = "NUMERIC"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"


# Ordem de abrangência entre os tipos numéricos inteiros (usada em
# type_inference.combinar_tipos). NUMERIC fica fora desta escala porque
# qualquer inteiro cabe em NUMERIC, mas a volta não é verdadeira.
ORDEM_INTEIROS = {
    TargetType.SMALLINT: 0,
    TargetType.INTEGER: 1,
    TargetType.BIGINT: 2,
}


@dataclass
class ParsedValue:
    """Uma célula, nos três estados RAW -> NORMALIZED -> TYPED."""

    raw: str
    normalized: Optional[str]  # string decimal canônica, ex. "6000.50" — nunca float
    state: ParseState
    value_class: ValueClass = ValueClass.EMPTY
    notes: str = ""


@dataclass
class ColumnProfile:
    """Perfil estatístico de uma coluna, calculado de forma vetorizada."""

    name: str
    n_total: int = 0
    n_null: int = 0
    n_blank: int = 0
    n_distinct: int = 0
    min_len: int = 0
    max_len: int = 0
    fixed_width: bool = False

    pct_numeric: float = 0.0
    pct_integer: float = 0.0
    pct_decimal: float = 0.0
    pct_date: float = 0.0
    pct_boolean: float = 0.0

    has_thousands_sep_dot: bool = False
    has_decimal_comma: bool = False
    has_decimal_dot: bool = False
    has_currency_symbol: bool = False
    has_percent_symbol: bool = False
    has_alpha: bool = False
    has_leading_zero: bool = False
    digits_only_pct: float = 0.0

    sample_values: list = field(default_factory=list)

    @property
    def n_non_empty(self) -> int:
        return self.n_total - self.n_null - self.n_blank


@dataclass
class TypeDecision:
    """Resultado da árvore de decisão de tipo para uma coluna."""

    target_type: TargetType
    scale: Optional[int] = None
    precision: Optional[int] = None
    varchar_len: Optional[int] = None
    is_identifier: bool = False
    identifier_kind: Optional[str] = None  # "CPF" | "CNPJ" | "EAN" | None
    confidence: float = 0.0
    reason: str = ""
    column_convention: ColumnConvention = ColumnConvention.UNKNOWN


@dataclass
class QualityIssue:
    """Um valor que não pôde ser convertido para o tipo decidido da coluna."""

    row: int
    column: str
    raw_value: str
    reason: str


@dataclass
class Thresholds:
    """Limiares configuráveis de aceitação de sujeira, por coluna/arquivo.

    - conversao_minima: fração mínima de valores não-nulos que precisa "bater"
      com o tipo candidato para a coluna assumir esse tipo (ex.: 0.95 = 95%).
    - accept/warn: fração de valores que viraram NULL por não bater com o
      tipo JÁ DECIDIDO da coluna (pós-decisão) — usada pelo validation_engine
      pra dar o veredito de qualidade: 0% = STRICT_OK, até `accept` = ACCEPT,
      até `warn` = WARN, acima disso = BLOCK. (0% / <0,1% / 0,1%-1% / >1%.)
    """

    conversao_minima: float = 0.95
    accept: float = 0.001
    warn: float = 0.01
    resolve_unknown_convention_as_br: bool = True
