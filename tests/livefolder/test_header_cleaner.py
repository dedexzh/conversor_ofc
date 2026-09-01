from livefolder import header_cleaner as H


def test_limpa_acentos_espacos_e_maiuscula():
    assert H.clean_column_name(" COD_PDV ") == "cod_pdv"
    assert H.clean_column_name("Código Cliente") == "código_cliente"
    assert H.clean_column_name("Razão Social") == "razão_social"


def test_none_e_nan_viram_vazio():
    assert H.clean_column_name(None) == ""
    assert H.clean_column_name(float("nan")) == ""


def test_dedupe_com_sufixo_numerico():
    nomes = H.clean_and_dedupe_columns(["Codigo", "Codigo", "Codigo"])
    assert nomes == ["codigo", "codigo_2", "codigo_3"]


def test_coluna_sem_nome_vira_placeholder():
    nomes = H.clean_and_dedupe_columns(["", "Nome", None, "Unnamed: 3"])
    assert nomes[0] == "coluna_sem_nome_0"
    assert nomes[1] == "nome"
    assert nomes[2] == "coluna_sem_nome_2"
    assert nomes[3] == "coluna_sem_nome_3"


def test_nunca_perde_coluna_por_colisao():
    nomes = H.clean_and_dedupe_columns(["A", "a", "A "])
    assert len(nomes) == len(set(nomes)) == 3
