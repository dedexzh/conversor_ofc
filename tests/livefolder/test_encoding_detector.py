from livefolder import encoding_detector as E


def test_bom_utf8():
    dados = b"\xef\xbb\xbfnome;valor\n1;2\n"
    r = E.detect_encoding(dados)
    assert r.encoding == "utf-8-sig"
    assert r.confidence == 1.0


def test_utf8_genuino_com_acentos():
    dados = "nome;razão_social\nJoão;Café Ltda\n".encode("utf-8")
    r = E.detect_encoding(dados)
    assert r.encoding == "utf-8"


def test_ascii_puro_reporta_cp1252():
    dados = b"nome;valor\nJoao;100\n"
    r = E.detect_encoding(dados)
    assert r.encoding == "cp1252"


def test_cp1252_genuino():
    dados = "nome;razão_social\nJoão;Café Ltda\n".encode("cp1252")
    r = E.detect_encoding(dados)
    assert r.encoding == "cp1252"
    assert r.confidence >= 0.9


def test_latin1_fallback_nunca_lanca_excecao():
    dados = bytes([0x81, 0x8D, 0x8F, 0x90, 0x9D])  # bytes indefinidos em cp1252
    r = E.detect_encoding(dados)
    assert r.encoding in ("cp1252", "latin1")


def test_coherence_score_texto_limpo():
    assert E.coherence_score("texto normal sem problema") == 1.0


def test_coherence_score_com_replacement_chars():
    score = E.coherence_score("texto com �� ruído")
    assert score < 1.0
