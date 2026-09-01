import decimal

import polars as pl
from sqlalchemy import NUMERIC, VARCHAR

from livefolder import postgres_writer as pw
from livefolder.models import TargetType, TypeDecision


def test_to_pandas_for_insert_numeric_vira_decimal():
    """Requer pyarrow instalado (Polars.to_pandas() depende dele) — sem essa
    dependência, a ponte com o to_sql existente quebra silenciosamente só na
    integração real com o Postgres, não nos testes do motor isolado."""
    df = pl.DataFrame({"valor": ["1234.56", None], "qtd": [10, 20]})
    decisions = {
        "valor": TypeDecision(target_type=TargetType.NUMERIC, scale=2, precision=6),
        "qtd": TypeDecision(target_type=TargetType.INTEGER),
    }
    pdf = pw.to_pandas_for_insert(df, decisions)
    assert pdf["valor"][0] == decimal.Decimal("1234.56")
    assert pdf["valor"][1] is None
    assert pdf["qtd"][0] == 10


def test_sql_type_name_for_varchar_e_numeric():
    d1 = TypeDecision(target_type=TargetType.VARCHAR, varchar_len=14)
    assert pw.sql_type_name_for(d1) == "VARCHAR(14)"
    d2 = TypeDecision(target_type=TargetType.NUMERIC, precision=10, scale=2)
    assert pw.sql_type_name_for(d2) == "NUMERIC(10,2)"
    d3 = TypeDecision(target_type=TargetType.BIGINT)
    assert pw.sql_type_name_for(d3) == "BIGINT"


def test_normalize_reflected_type():
    assert pw.normalize_reflected_type("TIMESTAMP WITHOUT TIME ZONE") == "TIMESTAMP"
    assert pw.normalize_reflected_type("CHARACTER VARYING") == "VARCHAR"
    assert pw.normalize_reflected_type("NUMERIC(10,2)") == "NUMERIC"
    assert pw.normalize_reflected_type("BOOLEAN") == "BOOLEAN"
    assert pw.normalize_reflected_type("some_unknown_type") == "TEXT"


# ---- coluna_precisa_evoluir ----
# Achado real: um arquivo criou "preco_minimo" como NUMERIC(4,2); outro
# arquivo do MESMO relatório (mesma pasta/tabela, após o merge por pasta)
# tinha valores maiores (ex.: 999,99) que não cabem em NUMERIC(4,2). A
# decisão reconciliada já vinha com a precisão certa, mas a checagem antiga
# só comparava a CATEGORIA do tipo ("NUMERIC" == "NUMERIC") e nunca disparava
# o ALTER COLUMN — o INSERT seguinte estourava.

def test_coluna_precisa_evoluir_categoria_diferente():
    decisao = TypeDecision(target_type=TargetType.NUMERIC, precision=10, scale=2)
    assert pw.coluna_precisa_evoluir("INTEGER", decisao) is True


def test_coluna_precisa_evoluir_numeric_precisao_insuficiente():
    tipo_refletido = NUMERIC(precision=4, scale=2)
    decisao = TypeDecision(target_type=TargetType.NUMERIC, precision=5, scale=2)
    assert pw.coluna_precisa_evoluir(tipo_refletido, decisao) is True


def test_coluna_precisa_evoluir_numeric_ja_cabe():
    tipo_refletido = NUMERIC(precision=10, scale=2)
    decisao = TypeDecision(target_type=TargetType.NUMERIC, precision=5, scale=2)
    assert pw.coluna_precisa_evoluir(tipo_refletido, decisao) is False


def test_coluna_precisa_evoluir_varchar_largura_insuficiente():
    tipo_refletido = VARCHAR(length=50)
    decisao = TypeDecision(target_type=TargetType.VARCHAR, varchar_len=100)
    assert pw.coluna_precisa_evoluir(tipo_refletido, decisao) is True


def test_coluna_precisa_evoluir_varchar_ja_cabe():
    tipo_refletido = VARCHAR(length=100)
    decisao = TypeDecision(target_type=TargetType.VARCHAR, varchar_len=50)
    assert pw.coluna_precisa_evoluir(tipo_refletido, decisao) is False
