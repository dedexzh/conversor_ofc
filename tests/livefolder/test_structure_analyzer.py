import polars as pl

from livefolder import structure_analyzer as S


def test_remove_coluna_100_por_cento_vazia():
    df = pl.DataFrame({"a": ["1", "2"], "vazia": [None, None]})
    out, report = S.analyze_and_clean(df)
    assert "vazia" not in out.columns
    assert report.empty_columns == ["vazia"]


def test_remove_coluna_so_espacos():
    df = pl.DataFrame({"a": ["1", "2"], "vazia": ["  ", " "]})
    out, report = S.analyze_and_clean(df)
    assert "vazia" not in out.columns


def test_remove_linhas_total_subtotal_resumo():
    df = pl.DataFrame({
        "a": ["1", "2", "TOTAL", "SUBTOTAL", "Resumo", "3"],
        "b": ["x", "y", "999", "888", "777", "z"],
    })
    out, report = S.analyze_and_clean(df)
    assert out["a"].to_list() == ["1", "2", "3"]
    assert report.junk_rows_dropped == 3


def test_lixo_so_avaliado_nas_3_primeiras_colunas():
    df = pl.DataFrame({
        "a": ["1", "2"],
        "b": ["x", "y"],
        "c": ["z", "w"],
        "d": ["TOTAL", "n"],  # 4a coluna — não deve contar
    })
    out, report = S.analyze_and_clean(df)
    assert out.height == 2
    assert report.junk_rows_dropped == 0


def test_remove_linhas_totalmente_vazias():
    df = pl.DataFrame({"a": ["1", None, "3"], "b": ["x", None, "z"]})
    out, report = S.analyze_and_clean(df)
    assert out.height == 2
    assert report.fully_empty_rows_dropped == 1


def test_dataframe_vazio_nao_quebra():
    df = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Utf8, "b": pl.Utf8})
    out, report = S.analyze_and_clean(df)
    assert out.height == 0
