"""Guard local (não roda em CI/outra máquina): reprocessa os arquivos reais
da pasta monitorada com o motor novo e garante 0 exceção não tratada. Não
grava no Postgres. Não comita dado real (a pasta tem CNPJ/nome de cliente
de verdade) — só compara contagens agregadas.
"""

import os

import pytest

from livefolder import pipeline

PASTA = r"C:\Users\dedex\OneDrive - Anheuser-Busch InBev\testes\GP7 - RELATÓRIOS"

pytestmark = pytest.mark.skipif(not os.path.isdir(PASTA), reason="pasta real não existe nesta máquina")


def _listar_arquivos():
    arquivos = []
    for root, _dirs, files in os.walk(PASTA):
        for f in files:
            if os.path.splitext(f)[1].lower() in (".csv", ".xlsx", ".xls", ".xlsm", ".xlsb"):
                arquivos.append(os.path.join(root, f))
    return arquivos


def test_todos_os_arquivos_reais_processam_sem_excecao():
    """Todo arquivo real precisa processar sem estourar uma exceção — EXCETO
    o caso explicitamente esperado de arquivo genuinamente vazio (só
    cabeçalho, 0 linhas de dado), que é uma rejeição correta, não um bug."""
    arquivos = _listar_arquivos()
    assert arquivos, "esperava encontrar arquivos na pasta monitorada"

    falhas = []
    for caminho in arquivos:
        try:
            pipeline.process_file(caminho)
        except ValueError as e:
            if "vazio" in str(e).lower():
                continue  # rejeição esperada de arquivo sem linhas de dado
            falhas.append((os.path.basename(caminho), f"{type(e).__name__}: {e}"))
        except Exception as e:  # noqa: BLE001 — qualquer outra exceção É um bug
            falhas.append((os.path.basename(caminho), f"{type(e).__name__}: {e}"))

    assert not falhas, "arquivos com exceção não tratada:\n" + "\n".join(f"{n}: {m}" for n, m in falhas)
