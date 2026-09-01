import polars as pl

from livefolder.readers import csv_reader, excel_reader


def test_le_csv_basico(tmp_path):
    caminho = tmp_path / "basico.csv"
    caminho.write_bytes("codigo;nome;valor\n0006000;A;1.234,56\n0006001;B;500,00\n".encode("cp1252"))
    df, diag = csv_reader.read_csv(str(caminho))
    assert df.height == 2
    assert df.width == 3
    assert diag.delimiter == ";"
    assert df["codigo"].to_list() == ["0006000", "0006001"]  # preserva zero à esquerda, tudo string


def test_le_csv_encoding_latin1(tmp_path):
    caminho = tmp_path / "latin1.csv"
    caminho.write_bytes("nome;cidade\nJoão;São Paulo\n".encode("latin1"))
    df, diag = csv_reader.read_csv(str(caminho))
    assert "São Paulo" in df["cidade"].to_list()


def test_le_csv_linha_com_colunas_extras_nao_quebra(tmp_path):
    caminho = tmp_path / "ragged.csv"
    caminho.write_bytes(b"a;b;c\n1;2;3\n4;5;6;7\n8;9\n")
    df, _diag = csv_reader.read_csv(str(caminho))
    assert df.height == 3  # não quebrou o arquivo inteiro


def test_le_csv_cabecalho_duplicado_nao_perde_coluna(tmp_path):
    """O próprio Polars já renomeia cabeçalhos duplicados na leitura (nunca
    perde uma coluna); header_cleaner então limpa esses nomes já únicos."""
    from livefolder import header_cleaner

    caminho = tmp_path / "dup.csv"
    caminho.write_bytes(b"codigo;codigo;valor\n1;2;3\n")
    df, _diag = csv_reader.read_csv(str(caminho))
    assert df.width == 3
    novos = header_cleaner.clean_and_dedupe_columns(df.columns)
    assert len(novos) == len(set(novos)) == 3
    assert df.select(pl.all()).row(0) == ("1", "2", "3")


def test_le_csv_campos_entre_aspas(tmp_path):
    caminho = tmp_path / "quoted.csv"
    caminho.write_bytes('nome;obs\n"SILVA, JOAO";"Cliente; importante"\n'.encode("utf-8"))
    df, _diag = csv_reader.read_csv(str(caminho))
    assert df.height == 1
    assert df["nome"][0] == "SILVA, JOAO"


def test_le_excel_basico(tmp_path):
    import openpyxl

    caminho = tmp_path / "basico.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["codigo", "valor"])
    ws.append(["0006000", "1.234,56"])
    wb.save(str(caminho))

    df, diag = excel_reader.read_excel(str(caminho))
    assert df.height == 1
    assert df["codigo"][0] == "0006000"
    assert diag.engine == "calamine"
    assert df.schema["codigo"] == pl.Utf8
