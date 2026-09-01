import polars as pl

from livefolder import pipeline
from livefolder.models import TargetType


def test_pipeline_arquivo_sujo_completo(tmp_path):
    """CSV sujo de ponta a ponta: cabeçalho duplicado, coluna vazia, linha
    TOTAL, zero à esquerda em quantidade real, decimal BR, CNPJ, data."""
    conteudo = (
        "Codigo;Codigo;Vazia;Quantidade;Valor;CNPJ;Data;Nome\n"
        "0006000;X;;0000112;1.234,56;11.222.333/0001-81;01/08/2026;Cliente A\n"
        "0006001;Y;;0000190;500,00;09.623.639/0001-06;02/08/2026;Cliente B\n"
        "TOTAL;;;;;;;\n"
    )
    caminho = tmp_path / "sujo.csv"
    caminho.write_bytes(conteudo.encode("cp1252"))

    resultado = pipeline.process_file(str(caminho))

    assert "vazia" not in resultado.df.columns
    assert resultado.structure_report.junk_rows_dropped == 1
    assert resultado.df.height == 2

    assert resultado.decisions["quantidade"].target_type in (
        TargetType.SMALLINT, TargetType.INTEGER, TargetType.BIGINT,
    )
    assert resultado.df["quantidade"].to_list() == [112, 190]

    assert resultado.decisions["valor"].target_type == TargetType.NUMERIC
    assert resultado.df["valor"].to_list() == ["1234.56", "500.00"]

    assert resultado.decisions["cnpj"].identifier_kind == "CNPJ"
    assert resultado.df["cnpj"].to_list() == ["11222333000181", "09623639000106"]

    assert resultado.decisions["data"].target_type == TargetType.DATE

    assert resultado.file_verdict.value in ("STRICT_OK", "ACCEPT")


def test_pipeline_conciliacao_com_padrao_global():
    """Se o padrão global já diz NUMERIC para uma coluna, um arquivo novo
    100% inteiro pra essa mesma coluna deve continuar NUMERIC (nunca regride
    pra INTEGER; a conciliação sempre anda pro tipo mais amplo)."""
    from livefolder.models import TypeDecision

    df1 = pl.DataFrame({"valor": ["6000", "7000"]})
    padrao = {"valor": TypeDecision(target_type=TargetType.NUMERIC, scale=2, precision=10, confidence=0.9)}
    resultado = pipeline.process_dataframe(df1, tipos_padrao_atual=padrao)
    assert resultado.decisions["valor"].target_type == TargetType.NUMERIC


def test_pipeline_tipo_fixo_da_tabela_nao_e_alargado_e_gera_anomalia():
    """Achado real: coluna 'vl_estouro_limite' fixada como NUMERIC(4,1) pela
    tabela (primeiro arquivo já carregado); um arquivo NOVO tem um valor bem
    maior ("13134.5"). `tipos_fixos` tem que VENCER — sem re-tipar/alargar
    (nada de INTEGER->NUMERIC ou NUMERIC mais largo aqui) — e o valor fora
    da faixa vira anomalia (NULL + issue), não um erro que derruba o
    arquivo inteiro."""
    from livefolder.models import TypeDecision

    df = pl.DataFrame({"valor": ["82.5", "13134.5"]})
    fixos = {"valor": TypeDecision(target_type=TargetType.NUMERIC, precision=4, scale=1, confidence=1.0)}
    resultado = pipeline.process_dataframe(df, tipos_fixos=fixos)
    assert resultado.decisions["valor"].target_type == TargetType.NUMERIC
    assert resultado.decisions["valor"].precision == 4
    assert resultado.decisions["valor"].scale == 1
    assert resultado.df["valor"].to_list() == ["82.5", None]
    assert len(resultado.issues["valor"]) == 1


def test_pipeline_arquivo_vazio_apos_limpeza_gera_dataframe_vazio():
    df = pl.DataFrame({"a": ["TOTAL"], "b": ["1"], "c": ["2"]})
    resultado = pipeline.process_dataframe(df)
    assert resultado.df.height == 0


def test_pipeline_numero_de_linha_da_anomalia_sobrevive_a_remocao_de_lixo():
    """Uma linha TOTAL removida ANTES do valor ruim não pode deslocar o
    número de linha reportado — precisa continuar apontando pra posição no
    arquivo ORIGINAL, não na posição pós-limpeza (Polars não preserva índice
    como o pandas fazia; ver _COL_INDICE_ORIGINAL em pipeline.py)."""
    valores_a = ["1", "TOTAL"] + [str(i) for i in range(2, 41)]
    valores_codigo = ["100", "999"] + (["ABC"] + [str(i) for i in range(3, 40)]) + ["200"]
    df = pl.DataFrame({"a": valores_a, "codigo": valores_codigo})
    resultado = pipeline.process_dataframe(df)
    assert resultado.structure_report.junk_rows_dropped == 1
    issues = resultado.issues["codigo"]
    assert len(issues) == 1
    # "ABC" está na linha 2 (0-based) do arquivo ORIGINAL (a linha TOTAL, no
    # índice 1, foi removida) — não na linha 1, que seria a posição dela
    # DEPOIS de remover a linha TOTAL.
    assert issues[0].row == 2
    assert issues[0].raw_value == "ABC"


def test_pipeline_poucas_linhas_sujas_nao_derruba_coluna_inteira():
    """Minoria de linha suja vira NULL + issue, não TEXT pra coluna toda."""
    valores = [str(i) for i in range(1, 200)] + ["LIXO"]
    df = pl.DataFrame({"codigo": valores})
    resultado = pipeline.process_dataframe(df)
    assert resultado.decisions["codigo"].target_type in (
        TargetType.SMALLINT, TargetType.INTEGER, TargetType.BIGINT,
    )
    assert len(resultado.issues["codigo"]) == 1
    assert resultado.issues["codigo"][0].raw_value == "LIXO"
