from livefolder import delimiter_detector as D


def test_ponto_virgula():
    texto = "codigo;nome;valor;data\n1;CLIENTE A;100,50;01/08/2026\n2;CLIENTE B;200,00;02/08/2026\n"
    r = D.detect_delimiter(texto)
    assert r.delimiter == ";"
    assert r.n_columns == 4


def test_virgula_nao_quebra_decimal_br_quando_delimitador_e_ponto_virgula():
    """CODIGO;NOME;VALOR;DATA com '100,50' não deve virar 5 colunas."""
    texto = "CODIGO;NOME;VALOR;DATA\n1;CLIENTE A;100,50;01/08/2026\n"
    r = D.detect_delimiter(texto)
    assert r.delimiter == ";"
    assert r.n_columns == 4


def test_virgula_como_delimitador():
    texto = "a,b,c\n1,2,3\n4,5,6\n"
    r = D.detect_delimiter(texto)
    assert r.delimiter == ","
    assert r.n_columns == 3


def test_tab():
    texto = "a\tb\tc\n1\t2\t3\n4\t5\t6\n"
    r = D.detect_delimiter(texto)
    assert r.delimiter == "\t"


def test_pipe():
    texto = "a|b|c\n1|2|3\n4|5|6\n"
    r = D.detect_delimiter(texto)
    assert r.delimiter == "|"


def test_delimitador_dentro_de_aspas_nao_conta():
    texto = 'nome;valor\n"SILVA, JOAO";100\n"PADRAO";200\n'
    r = D.detect_delimiter(texto)
    assert r.delimiter == ";"
    assert r.n_columns == 2


def test_arquivo_vazio_usa_padrao_br():
    r = D.detect_delimiter("")
    assert r.delimiter == ";"
    assert r.confidence == 0.0
