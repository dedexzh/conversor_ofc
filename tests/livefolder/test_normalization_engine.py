import polars as pl

from livefolder import normalization_engine as N
from livefolder.models import ColumnConvention, TargetType, TypeDecision


def test_integer_preserva_validos_e_marca_invalido():
    s = pl.Series("qtd", ["0000112", "0000190", "ABC", None, "0009117"])
    d = TypeDecision(target_type=TargetType.INTEGER, column_convention=ColumnConvention.BR)
    res, issues = N.normalize_column(s, d)
    assert res.to_list() == [112, 190, None, None, 9117]
    assert len(issues) == 1
    assert issues[0].raw_value == "ABC"
    assert issues[0].row == 2


def test_numeric_string_decimal_canonica():
    s = pl.Series("valor", ["1.234,56", "6,5", "xyz", ""])
    d = TypeDecision(target_type=TargetType.NUMERIC, column_convention=ColumnConvention.BR)
    res, issues = N.normalize_column(s, d)
    assert res.to_list() == ["1234.56", "6.5", None, None]
    assert len(issues) == 1  # só "xyz" — "" é ausência legítima, não erro
    assert issues[0].raw_value == "xyz"


def test_boolean_mapeia_vocabulario_conhecido():
    s = pl.Series("ativo", ["S", "N", "s", "talvez", None])
    d = TypeDecision(target_type=TargetType.BOOLEAN)
    res, issues = N.normalize_column(s, d)
    assert res.to_list() == [True, False, True, None, None]
    assert len(issues) == 1
    assert issues[0].raw_value == "talvez"


def test_date_produz_datas():
    s = pl.Series("data", ["01/08/2026", "02/08/2026", "invalido", None])
    d = TypeDecision(target_type=TargetType.DATE)
    res, issues = N.normalize_column(s, d)
    import datetime
    assert res.to_list() == [datetime.date(2026, 8, 1), datetime.date(2026, 8, 2), None, None]
    assert len(issues) == 1


def test_timestamp_preserva_hora():
    s = pl.Series("dh", ["01/08/2026 14:30:00"])
    d = TypeDecision(target_type=TargetType.TIMESTAMP)
    res, _issues = N.normalize_column(s, d)
    v = res.to_list()[0]
    assert v.hour == 14 and v.minute == 30


def test_identifier_normaliza_para_so_digitos():
    s = pl.Series("cnpj", ["11.222.333/0001-81", "09.623.639/0001-06"])
    d = TypeDecision(target_type=TargetType.VARCHAR, is_identifier=True, identifier_kind="CNPJ", varchar_len=14)
    res, issues = N.normalize_column(s, d)
    assert res.to_list() == ["11222333000181", "09623639000106"]
    assert issues == []


def test_texto_generico_so_faz_strip():
    s = pl.Series("nome", ["  Coca Cola  ", "Brahma"])
    d = TypeDecision(target_type=TargetType.TEXT)
    res, issues = N.normalize_column(s, d)
    assert res.to_list() == ["Coca Cola", "Brahma"]
    assert issues == []


def test_numeric_com_precisao_fixa_valor_fora_da_faixa_vira_anomalia():
    """Achado real: schema de tabela FIXO em NUMERIC(4,1) (3 dígitos
    inteiros); um arquivo novo traz um valor com 5 dígitos inteiros
    ("13134.5"). Sem essa checagem, o valor passava reto e só estourava no
    INSERT do Postgres, derrubando o arquivo inteiro — agora vira anomalia
    (NULL + log), preservando o resto do arquivo."""
    s = pl.Series("valor", ["82.5", "13134.5", None])
    d = TypeDecision(target_type=TargetType.NUMERIC, precision=4, scale=1, column_convention=ColumnConvention.BR)
    res, issues = N.normalize_column(s, d)
    assert res.to_list() == ["82.5", None, None]
    assert len(issues) == 1
    assert issues[0].raw_value == "13134.5"
    assert "NUMERIC(4,1)" in issues[0].reason


def test_numeric_com_precisao_fixa_zero_a_esquerda_nao_conta_como_digito():
    """Achado real: export de largura fixa zero-padded ('00000000000,00',
    11 caracteres) contra um tipo fixo NUMERIC(11,1) (10 dígitos inteiros)
    NÃO pode ser rejeitado — o valor É zero, não tem 11 dígitos de verdade.
    Contar o comprimento bruto da string (sem descartar zero à esquerda)
    rejeitava valores pequenos só por causa do padding."""
    s = pl.Series("valor", [" 00000000000,00", " 00000075165,00"])
    d = TypeDecision(target_type=TargetType.NUMERIC, precision=11, scale=1, column_convention=ColumnConvention.BR)
    res, issues = N.normalize_column(s, d)
    assert issues == []
    assert res.to_list() == ["00000000000.00", "00000075165.00"]


def test_varchar_com_largura_fixa_valor_maior_vira_anomalia():
    s = pl.Series("nome", ["Bar Curto", "Um Nome de Bar Extremamente Longo Que Nao Cabe"])
    d = TypeDecision(target_type=TargetType.VARCHAR, varchar_len=14)
    res, issues = N.normalize_column(s, d)
    assert res.to_list() == ["Bar Curto", None]
    assert len(issues) == 1
    assert "VARCHAR(14)" in issues[0].reason


def test_null_original_nao_vira_issue():
    """Um valor originalmente nulo/vazio nunca é um 'erro de conversão' —
    é ausência de dado, tratado separado de valor inválido."""
    s = pl.Series("qtd", [None, "", "  "])
    d = TypeDecision(target_type=TargetType.INTEGER, column_convention=ColumnConvention.BR)
    _res, issues = N.normalize_column(s, d)
    assert issues == []
