from livefolder import validation_engine as V
from livefolder.models import Thresholds


def test_zero_erros_strict_ok():
    assert V.evaluate_column(1000, 0) == V.Verdict.STRICT_OK


def test_taxa_baixa_accept():
    # 0.05% < accept (0.1%)
    assert V.evaluate_column(2000, 1) == V.Verdict.ACCEPT


def test_taxa_media_warn():
    # 0.5%, entre accept (0.1%) e warn (1%)
    assert V.evaluate_column(1000, 5) == V.Verdict.WARN


def test_taxa_alta_block():
    # 5%, acima de warn (1%)
    assert V.evaluate_column(1000, 50) == V.Verdict.BLOCK


def test_limiares_customizados():
    thresholds = Thresholds(accept=0.05, warn=0.20)
    assert V.evaluate_column(100, 3, thresholds) == V.Verdict.ACCEPT  # 3%
    assert V.evaluate_column(100, 15, thresholds) == V.Verdict.WARN   # 15%
    assert V.evaluate_column(100, 30, thresholds) == V.Verdict.BLOCK  # 30%


def test_evaluate_file_pega_pior_veredito():
    vereditos = {"a": V.Verdict.STRICT_OK, "b": V.Verdict.WARN, "c": V.Verdict.ACCEPT}
    assert V.evaluate_file(vereditos) == V.Verdict.WARN


def test_evaluate_file_vazio():
    assert V.evaluate_file({}) == V.Verdict.STRICT_OK


def test_status_para_relatorio():
    assert V.status_para_relatorio(V.Verdict.STRICT_OK) == "SUCCESS"
    assert V.status_para_relatorio(V.Verdict.ACCEPT) == "SUCCESS"
    assert V.status_para_relatorio(V.Verdict.WARN) == "SUCCESS_WITH_WARNINGS"
    assert V.status_para_relatorio(V.Verdict.BLOCK) == "BLOCKED"
