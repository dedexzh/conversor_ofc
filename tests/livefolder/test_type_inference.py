import polars as pl

from livefolder import data_profiler, type_inference as TI
from livefolder.models import TargetType, Thresholds, TypeDecision


def _infer(values, name_hint=None, thresholds=None):
    s = pl.Series("v", values)
    profile = data_profiler.profile_column(s, thresholds)
    stripped = s.cast(pl.Utf8, strict=False).str.strip_chars()
    non_empty = stripped.filter(s.is_not_null() & (stripped != ""))
    return TI.infer_column_type(profile, non_empty, name_hint=name_hint, thresholds=thresholds)


def test_bandeira_zero_a_esquerda_converge():
    """0006000 / 6000 / 6.000, cada um "vindo de um arquivo diferente" (mas
    tipados isoladamente aqui), devem produzir o MESMO tipo alvo."""
    tipos = {_infer([v] * 20).target_type for v in ["0006000", "6000", "6.000"]}
    assert tipos == {TargetType.SMALLINT}


def test_cliente_largura_fixa_zero_esquerda_vira_numerico():
    """Confere contra o catálogo real (tipos_de_dados.yml do Power Query):
    'cliente' fixo em 7 dígitos com zero à esquerda é NUMERIC lá — bate."""
    valores = [f"{i:07d}" for i in range(1, 50)]
    d = _infer(valores, name_hint="cliente")
    assert d.target_type in (TargetType.SMALLINT, TargetType.INTEGER, TargetType.BIGINT)
    assert not d.is_identifier


def test_largura_variavel_alta_cardinalidade_fica_texto():
    valores = []
    for i in range(1, 101):
        valores.append(f"0{i}" if i % 3 == 0 else (f"00{i}" if i % 3 == 1 else str(i)))
    d = _infer(valores)
    assert d.target_type == TargetType.TEXT
    assert d.is_identifier


def test_cpf_valido_por_checksum():
    d = _infer(["111.444.777-35"] * 30)
    assert d.target_type == TargetType.VARCHAR
    assert d.identifier_kind == "CPF"
    assert d.varchar_len == 11


def test_cnpj_valido_por_checksum():
    d = _infer(["11.222.333/0001-81"] * 30)
    assert d.identifier_kind == "CNPJ"
    assert d.varchar_len == 14


def test_ean_valido_por_checksum():
    d = _infer(["4006381333931"] * 30)
    assert d.identifier_kind == "EAN"


def test_cnpj_maioria_invalida_ainda_e_identificador():
    """Achado real: uma coluna 'cnpj' de cadastro de clientes real teve só
    29% de checksum válido (placeholders/erros de digitação são comuns em
    cadastro real), mas 100% de comprimento 14 — ainda assim é claramente
    uma coluna de CNPJ (29% é ~35x o que puro acaso produziria para 2
    dígitos verificadores, ~0,8%) e precisa continuar VARCHAR(14), não virar
    BIGINT e perder o zero à esquerda."""
    validos = ["11.222.333/0001-81"] * 3  # ~29% válidos, resto malformado mas com 14 dígitos
    invalidos = [f"{i:014d}" for i in range(1, 8)]  # 14 dígitos, checksum quase certamente inválido
    d = _infer((validos + invalidos) * 5)
    assert d.identifier_kind == "CNPJ"
    assert d.target_type == TargetType.VARCHAR
    assert d.varchar_len == 14


def test_ean_precisa_de_taxa_bem_acima_de_10_por_cento():
    """EAN só tem 1 dígito verificador (~10% de acerto por puro acaso) — uma
    coluna numérica de 13 dígitos qualquer (não EAN de verdade) não pode virar
    identificador só por coincidência estatística baixa."""
    # sequência crescente de 13 dígitos — não são EAN válidos de propósito,
    # só têm o comprimento certo; taxa de acerto deve ficar perto do acaso.
    valores = [f"{1000000000000 + i}" for i in range(1, 51)]
    d = _infer(valores)
    assert d.identifier_kind != "EAN"


def test_nome_nao_e_sinal_primario_regressao():
    """Coluna chamada 'cpf' mas com inteiros limpos que NÃO batem checksum de
    CPF de verdade: precisa virar INTEGER/BIGINT mesmo assim — prova que o
    nome não é mais o sinal primário (era o bug corrigido nesta sessão)."""
    valores = [str(i) for i in range(100000, 100050)]
    d = _infer(valores, name_hint="cpf")
    assert d.target_type in (TargetType.SMALLINT, TargetType.INTEGER, TargetType.BIGINT)
    assert not d.is_identifier
    assert d.identifier_kind is None


def test_booleano_s_n():
    d = _infer(["S", "N", "S", "S", "N"] * 10)
    assert d.target_type == TargetType.BOOLEAN


def test_booleano_sim_nao():
    d = _infer(["SIM", "NAO", "sim", "nao"] * 10)
    assert d.target_type == TargetType.BOOLEAN


def test_zero_um_sem_nome_sugestivo_fica_numerico():
    d = _infer(["0", "1", "0", "1"] * 10, name_hint="quantidade")
    assert d.target_type != TargetType.BOOLEAN


def test_zero_um_com_nome_sugestivo_vira_booleano():
    d = _infer(["0", "1"] * 10, name_hint="flag_ativo")
    assert d.target_type == TargetType.BOOLEAN


def test_data_sem_hora_vira_date():
    d = _infer(["01/08/2026", "02/08/2026", "15/12/2025"] * 10)
    assert d.target_type == TargetType.DATE


def test_data_com_hora_vira_timestamp():
    d = _infer(["01/08/2026 14:30:00", "02/08/2026 09:15:22"] * 10)
    assert d.target_type == TargetType.TIMESTAMP


def test_decimal_com_virgula_numeric_escala():
    d = _infer(["1.234,56", "6,5", "100,00"] * 10)
    assert d.target_type == TargetType.NUMERIC
    assert d.scale == 2


def test_texto_puro():
    d = _infer(["Coca Cola", "Brahma", "Guaraná Antarctica"] * 10)
    assert d.target_type in (TargetType.VARCHAR, TargetType.TEXT)


def test_coluna_vazia():
    d = _infer([None, "", "  ", None])
    assert d.target_type == TargetType.TEXT
    assert d.confidence == 0.0


def test_convencao_conflitante_vira_texto():
    d = _infer(["1.234,56", "6.5"] * 10)
    assert d.target_type == TargetType.TEXT


def test_convencao_ambigua_estrita_vira_texto():
    thresholds = Thresholds(resolve_unknown_convention_as_br=False)
    d = _infer(["6.000", "1.234"] * 10, thresholds=thresholds)
    assert d.target_type == TargetType.TEXT


def test_inteiro_maior_que_bigint_vira_numeric():
    grande = "9" * 25  # 25 dígitos, muito maior que BIGINT
    d = _infer([grande] * 10)
    assert d.target_type == TargetType.NUMERIC
    assert d.scale == 0


def test_decidir_tipo_numerico_precisao_cabe_maior_inteiro_e_maior_escala():
    # Achado real: "5844.9977" (4 casas decimais) e "13134.532" (5 dígitos
    # inteiros, só 3 casas decimais) têm o MESMO nº de dígitos totais (8) —
    # mas a coluna INTEIRA é escalada pela MAIOR escala vista (4), então
    # precisa caber os 5 dígitos inteiros de "13134.532" JUNTO com essas 4
    # casas decimais (9 no total). Usar só o maior comprimento total de cada
    # valor isolado (8) cortava um dígito inteiro e estourava no INSERT.
    normalized = pl.Series(["5844.9977", "13134.532", "65.4383"])
    d = TI._decidir_tipo_numerico(normalized)
    assert d.target_type == TargetType.NUMERIC
    assert d.scale == 4
    assert d.precision == 9  # 5 dígitos inteiros (13134) + 4 casas decimais


# ---- combinar_tipos ----

def test_combinar_tipos_escalona_inteiros():
    a = TypeDecision(target_type=TargetType.INTEGER, confidence=0.9)
    b = TypeDecision(target_type=TargetType.BIGINT, confidence=0.9)
    assert TI.combinar_tipos(a, b).target_type == TargetType.BIGINT
    assert TI.combinar_tipos(b, a).target_type == TargetType.BIGINT


def test_combinar_tipos_text_nunca_regride():
    a = TypeDecision(target_type=TargetType.TEXT, confidence=0.9)
    b = TypeDecision(target_type=TargetType.INTEGER, confidence=0.9)
    assert TI.combinar_tipos(a, b).target_type == TargetType.TEXT
    assert TI.combinar_tipos(b, a).target_type == TargetType.TEXT


def test_combinar_tipos_inteiro_com_numeric_vira_numeric():
    a = TypeDecision(target_type=TargetType.INTEGER, confidence=0.9)
    b = TypeDecision(target_type=TargetType.NUMERIC, scale=2, precision=10, confidence=0.9)
    assert TI.combinar_tipos(a, b).target_type == TargetType.NUMERIC
    assert TI.combinar_tipos(b, a).target_type == TargetType.NUMERIC


def test_combinar_tipos_inteiro_com_numeric_amplia_precisao_para_caber_o_inteiro():
    # Achado real: coluna "vl_estouro_limite" virou INTEGER num arquivo (ex.:
    # UNB 817260, só valores inteiros de até 4 dígitos já gravados na
    # tabela). Outro arquivo do MESMO relatório (UNB 835242) tem casas
    # decimais mas só amostrou valores pequenos -> NUMERIC(4,1). Conciliar
    # direto por NUMERIC(4,1) (só 3 dígitos inteiros) faz o ALTER COLUMN
    # subsequente estourar ao converter os inteiros de 4 dígitos já gravados
    # pelo outro arquivo — por isso a precisão final tem que caber a faixa
    # MÁXIMA que o tipo inteiro (INTEGER) permite, não só a amostra atual.
    inteiro = TypeDecision(target_type=TargetType.INTEGER, confidence=0.9)
    numerico_amostra_pequena = TypeDecision(target_type=TargetType.NUMERIC, scale=1, precision=4, confidence=0.9)
    r = TI.combinar_tipos(inteiro, numerico_amostra_pequena)
    assert r.target_type == TargetType.NUMERIC
    assert r.scale == 1
    assert r.precision >= 11  # 10 dígitos (range do INTEGER) + escala 1
    # Mesmo resultado independente da ordem (padrão global vs. arquivo atual).
    r_invertido = TI.combinar_tipos(numerico_amostra_pequena, inteiro)
    assert r_invertido.precision == r.precision


def test_combinar_tipos_date_com_timestamp_vira_timestamp():
    a = TypeDecision(target_type=TargetType.DATE, confidence=0.9)
    b = TypeDecision(target_type=TargetType.TIMESTAMP, confidence=0.9)
    assert TI.combinar_tipos(a, b).target_type == TargetType.TIMESTAMP


def test_combinar_tipos_mesmo_numeric_usa_maior_escala():
    # a = NUMERIC(8,2) -> 6 dígitos inteiros; b = NUMERIC(6,4) -> 2 dígitos
    # inteiros. O resultado precisa caber os 6 dígitos inteiros de "a" MAIS a
    # escala 4 de "b" (10), não só maxar precisão/escala independentemente
    # (que daria 8 e cortaria os dígitos inteiros de "a" — achado real, ver
    # comentário em combinar_tipos).
    a = TypeDecision(target_type=TargetType.NUMERIC, scale=2, precision=8, confidence=0.9)
    b = TypeDecision(target_type=TargetType.NUMERIC, scale=4, precision=6, confidence=0.9)
    r = TI.combinar_tipos(a, b)
    assert r.scale == 4
    assert r.precision == 10


def test_combinar_tipos_none_anterior_retorna_nova():
    b = TypeDecision(target_type=TargetType.BIGINT, confidence=0.9)
    assert TI.combinar_tipos(None, b) is b


def test_combinar_tipos_incompativel_cai_para_text():
    a = TypeDecision(target_type=TargetType.BOOLEAN, confidence=0.9)
    b = TypeDecision(target_type=TargetType.DATE, confidence=0.9)
    assert TI.combinar_tipos(a, b).target_type == TargetType.TEXT
