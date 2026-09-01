import polars as pl
import pytest

from livefolder import ptbr_parser as P
from livefolder.models import ColumnConvention, ParseState, Thresholds


CASOS_INEQUIVOCOS = [
    ("6", "6"),
    ("6,5", "6.5"),
    ("6,50", "6.50"),
    ("6.000", "6000"),
    ("6.000,50", "6000.50"),
    ("1.250.000,75", "1250000.75"),
    ("0006000", "6000"),
    ("00006000", "6000"),
    ("-6.000,50", "-6000.50"),
    ("R$ 6.000,00", "6000.00"),
    ("6,00", "6.00"),
    ("6.000,00", "6000.00"),
    (" 6000 ", "6000"),
    ("(1.234,56)", "-1234.56"),
    # Achado real: export de largura fixa zero-padded, SEM separador de
    # milhar nenhum (11 dígitos direto antes da vírgula) — antes caía em
    # MALFORMED (a regex só aceitava 1-3 dígitos soltos ou grupos de
    # exatamente 3 separados por ponto) e virava NULL mesmo sendo BR válido.
    (" 00000075165,00", "00000075165.00"),
]


@pytest.mark.parametrize("raw,esperado", CASOS_INEQUIVOCOS)
def test_parse_values_pure_python(raw, esperado):
    parsed, _convention = P.parse_values([raw])
    assert parsed[0].normalized == esperado
    assert parsed[0].state == ParseState.NORMALIZED


def test_parse_values_pure_python_batch_convergem():
    """0006000 / 6000 / 6.000 (three different source-file styles) devem
    convergir pro MESMO valor normalizado quando lidos juntos como coluna."""
    parsed, convention = P.parse_values(["0006000", "6000", "6.000"])
    assert convention == ColumnConvention.BR
    assert [p.normalized for p in parsed] == ["6000", "6000", "6000"]


def test_vetorizado_bate_com_pure_python():
    raws = [c[0] for c in CASOS_INEQUIVOCOS]
    esperados = [c[1] for c in CASOS_INEQUIVOCOS]

    parsed, _ = P.parse_values(raws)
    assert [p.normalized for p in parsed] == esperados

    df = pl.DataFrame({"v": raws})
    shapes = df.select(P.classify_series(pl.col("v")).alias("s"))["s"]
    convention = P.infer_column_convention_from_series(shapes)
    out = df.select(P.parse_series(pl.col("v"), convention).alias("n"))["n"].to_list()
    assert out == esperados


def test_convencao_iso_por_evidencia():
    """Ponto com dígitos != 3 em QUALQUER valor da coluna prova ISO — mesmo
    que outro valor da coluna seja isoladamente ambíguo (6.000)."""
    parsed, convention = P.parse_values(["6.5", "1234.56", "6.000"])
    assert convention == ColumnConvention.ISO
    assert parsed[2].normalized == "6.000"  # NÃO removeu o ponto (ISO)


def test_convencao_br_por_evidencia_de_virgula():
    parsed, convention = P.parse_values(["1.234,56", "6.000"])
    assert convention == ColumnConvention.BR
    assert parsed[1].normalized == "6000"


def test_unknown_estrito_marca_ambiguous():
    thresholds = Thresholds(resolve_unknown_convention_as_br=False)
    parsed, convention = P.parse_values(["6.000", "1.234"], thresholds)
    assert convention == ColumnConvention.UNKNOWN
    assert all(p.state == ParseState.AMBIGUOUS for p in parsed)
    assert all(p.normalized is None for p in parsed)


def test_unknown_default_resolve_br():
    thresholds = Thresholds(resolve_unknown_convention_as_br=True)
    parsed, convention = P.parse_values(["6.000", "1.234"], thresholds)
    assert convention == ColumnConvention.BR
    assert [p.normalized for p in parsed] == ["6000", "1234"]


def test_conflict_br_e_iso_juntos():
    _parsed, convention = P.parse_values(["1.234,56", "6.5"])
    assert convention == ColumnConvention.CONFLICT


def test_valor_nao_numerico():
    parsed, _ = P.parse_values(["ABC123"])
    assert parsed[0].state == ParseState.INVALID


def test_valor_vazio_e_nulo():
    parsed, _ = P.parse_values([None, "", "  ", "nan"])
    assert all(p.state == ParseState.EMPTY for p in parsed)


def test_malformado_multiplas_virgulas():
    parsed, _ = P.parse_values(["1,2,3"])
    assert parsed[0].state == ParseState.INVALID


def test_classify_value_shape_casos_basicos():
    from livefolder.models import ValueClass

    assert P.classify_value_shape("6000")[0] == ValueClass.INTEGER_PLAIN
    assert P.classify_value_shape("6,5")[0] == ValueClass.DECIMAL_COMMA
    assert P.classify_value_shape("1.250.000")[0] == ValueClass.MULTI_DOT_GROUPED
    assert P.classify_value_shape("6.5")[0] == ValueClass.SINGLE_DOT_OTHER_DIGITS
    assert P.classify_value_shape("6.000")[0] == ValueClass.SINGLE_DOT_3DIGITS
    assert P.classify_value_shape("ABC")[0] == ValueClass.NOT_NUMERIC_SHAPED
    assert P.classify_value_shape(None)[0] == ValueClass.EMPTY
