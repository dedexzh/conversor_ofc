"""Ponte entre o motor novo (Polars) e a escrita em Postgres que já existe e
já funciona: `DatabaseManager`/`to_sql`/evolução de schema continuam em
`conversor_db.py`, inalterados — aqui só fica a conversão Polars -> pandas
(último passo antes do `to_sql`) e o mapeamento de `TargetType` pro
SQLAlchemy. `DatabaseManager` não é importado aqui de propósito, pra evitar
import circular (`conversor_db.py` importa `livefolder`, não o contrário).

Tabelas novas desenhadas para a Fase 3 (relatório de qualidade + alerta de
schema — NÃO criadas nesta rodada):

    CREATE TABLE tb_relatorio_qualidade (
        id SERIAL PRIMARY KEY, caminho_arquivo TEXT, nome_arquivo VARCHAR(255),
        tabela_destino VARCHAR(100), data_processamento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        encoding_detectado VARCHAR(20), encoding_confianca FLOAT, delimitador_detectado VARCHAR(5),
        linhas_totais INT, colunas_originais INT, colunas_validas INT, colunas_vazias INT,
        linhas_ignoradas_estrutura INT, valores_normalizados INT, valores_invalidos INT,
        valores_ambiguos INT, status VARCHAR(30), detalhe_json JSONB
    );
    CREATE TABLE tb_qualidade_colunas (
        id SERIAL PRIMARY KEY, relatorio_id INT REFERENCES tb_relatorio_qualidade(id) ON DELETE CASCADE,
        coluna VARCHAR(100), tipo_inferido VARCHAR(30), precisao INT, escala INT,
        is_identificador BOOLEAN, tipo_identificador VARCHAR(20), convencao_numerica VARCHAR(10),
        pct_nulo FLOAT, pct_valido FLOAT, confianca FLOAT
    );
    CREATE TABLE tb_schema_change_log (
        id SERIAL PRIMARY KEY, tabela VARCHAR(100), coluna VARCHAR(100),
        tipo_anterior VARCHAR(30), tipo_novo VARCHAR(30), caminho_arquivo TEXT,
        data_evento TIMESTAMP DEFAULT CURRENT_TIMESTAMP, severidade VARCHAR(20),
        aplicado BOOLEAN, detalhe TEXT
    );
"""

from __future__ import annotations

import decimal

import pandas as pd
import polars as pl
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Integer, Numeric, SmallInteger, String, Text

from .models import TargetType, TypeDecision

_TIPO_SQLALCHEMY_SIMPLES = {
    TargetType.BOOLEAN: Boolean,
    TargetType.SMALLINT: SmallInteger,
    TargetType.INTEGER: Integer,
    TargetType.BIGINT: BigInteger,
    TargetType.DATE: Date,
    TargetType.TIMESTAMP: DateTime,
    TargetType.TEXT: Text,
}


def sqlalchemy_type_for(decision: TypeDecision):
    """Tipo SQLAlchemy correspondente à decisão — usado no `dtype=` do
    `to_sql` e na evolução de schema (`ALTER TABLE ... TYPE ...`)."""
    if decision.target_type == TargetType.VARCHAR:
        return String(decision.varchar_len or 255)
    if decision.target_type == TargetType.NUMERIC:
        if decision.precision:
            return Numeric(decision.precision, decision.scale or 0)
        return Numeric
    return _TIPO_SQLALCHEMY_SIMPLES[decision.target_type]


def sql_type_name_for(decision: TypeDecision) -> str:
    """Nome literal do tipo Postgres (pra DDL bruto: `ADD COLUMN`/`ALTER
    COLUMN ... TYPE`) — distinto de `sqlalchemy_type_for`, que devolve um
    objeto de tipo SQLAlchemy pro `dtype=` do `to_sql`."""
    if decision.target_type == TargetType.VARCHAR:
        return f"VARCHAR({decision.varchar_len or 255})"
    if decision.target_type == TargetType.NUMERIC:
        if decision.precision:
            return f"NUMERIC({decision.precision},{decision.scale or 0})"
        return "NUMERIC"
    return decision.target_type.value  # BOOLEAN/SMALLINT/INTEGER/BIGINT/DATE/TIMESTAMP/TEXT já são nomes Postgres válidos


def normalize_reflected_type(sql_type) -> str:
    """Traduz o tipo de coluna refletido do banco (ex.: 'TIMESTAMP WITHOUT
    TIME ZONE', 'CHARACTER VARYING') para um dos rótulos de `TargetType`,
    permitindo comparar o que já existe na tabela com a decisão atual."""
    nome = str(sql_type).upper()
    if nome.startswith("BOOL"):
        return TargetType.BOOLEAN.value
    if nome.startswith("SMALLINT"):
        return TargetType.SMALLINT.value
    if nome.startswith("BIGINT"):
        return TargetType.BIGINT.value
    if nome.startswith("INTEGER") or nome.startswith("INT4") or nome == "INT":
        return TargetType.INTEGER.value
    if nome.startswith("NUMERIC") or nome.startswith("DECIMAL"):
        return TargetType.NUMERIC.value
    if nome.startswith("TIMESTAMP"):
        return TargetType.TIMESTAMP.value
    if nome.startswith("DATE"):
        return TargetType.DATE.value
    if nome.startswith("VARCHAR") or nome.startswith("CHARACTER VARYING"):
        return TargetType.VARCHAR.value
    return TargetType.TEXT.value


def coluna_precisa_evoluir(tipo_refletido, decisao: TypeDecision) -> bool:
    """True se a coluna JÁ EXISTENTE na tabela precisa de `ALTER COLUMN
    TYPE` para caber a decisão atual — não só quando a CATEGORIA de tipo
    mudou (ex.: INTEGER -> NUMERIC, já coberto por `normalize_reflected_type`
    ), mas também quando a categoria já bate e só a LARGURA que não é
    suficiente (ex.: NUMERIC(4,2) já gravado na tabela, decisão atual pede
    NUMERIC(5,2); VARCHAR(50) já gravado, decisão atual pede VARCHAR(100)).

    Sem isso, duas colunas rotuladas 'NUMERIC' (ou 'VARCHAR') pareciam
    idênticas mesmo quando uma delas não cabia o valor da outra, e o
    ALTER COLUMN nunca rodava — o INSERT seguinte estourava (achado real:
    'numeric field overflow' em 'preco_minimo' com um arquivo que tinha
    valores maiores, mesmo já mesclando na tabela certa)."""
    categoria_atual = normalize_reflected_type(tipo_refletido)
    if categoria_atual != decisao.target_type.value:
        return True
    if decisao.target_type == TargetType.NUMERIC:
        precisao_atual = getattr(tipo_refletido, "precision", None) or 0
        escala_atual = getattr(tipo_refletido, "scale", None) or 0
        return (decisao.precision or 0) > precisao_atual or (decisao.scale or 0) > escala_atual
    if decisao.target_type == TargetType.VARCHAR:
        largura_atual = getattr(tipo_refletido, "length", None) or 0
        return (decisao.varchar_len or 0) > largura_atual
    return False


def to_pandas_for_insert(df: pl.DataFrame, decisions: dict[str, TypeDecision]) -> pd.DataFrame:
    """Converte o DataFrame Polars final para pandas, no último passo antes
    do `to_sql` existente. Colunas NUMERIC viram `decimal.Decimal` (nunca
    `float`) — preserva precisão exata em campos financeiros; o resto é
    conversão direta."""
    pdf = df.to_pandas()
    for nome, decisao in decisions.items():
        if nome not in pdf.columns or decisao.target_type != TargetType.NUMERIC:
            continue
        # to_pandas() representa nulo de uma coluna string como float('nan')
        # (via pyarrow), não como None — "v is not None" sozinho deixava passar
        # e virava Decimal('NaN') em vez de NULL de verdade na coluna Postgres.
        pdf[nome] = pdf[nome].map(lambda v: None if pd.isna(v) else decimal.Decimal(v))
    return pdf


def is_regression(tipo_anterior: str, tipo_novo: str) -> bool:
    """True se a mudança de tipo parece uma REGRESSÃO (perda de tipagem, ex.:
    NUMERIC virando TEXT) em vez de evolução esperada. Não é usada nesta
    rodada (fica pronta para o alerta de schema da Fase 3)."""
    return tipo_anterior != tipo_novo and tipo_novo == "TEXT" and tipo_anterior != "TEXT"
