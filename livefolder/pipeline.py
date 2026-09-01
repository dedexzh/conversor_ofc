"""Orquestra o pipeline completo (Fases A-C do parser + profiling + decisão
de tipo + normalização + validação) sobre um DataFrame já lido. É o ÚNICO
ponto de integração que `conversor_db.py` precisa chamar — mantém o
`IngestionEngine.processar_arquivo` fino, sem reimplementar a lógica aqui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional

import polars as pl

from . import data_profiler, header_cleaner, normalization_engine, ptbr_parser, structure_analyzer, type_inference
from . import validation_engine as ve
from .models import ColumnConvention, ColumnProfile, QualityIssue, Thresholds, TypeDecision
from .readers import csv_reader, excel_reader
from .structure_analyzer import StructureReport

EXTENSOES_EXCEL = excel_reader.EXTENSOES_SUPORTADAS

# Nome interno da coluna de rastreio de linha original — Polars (diferente do
# pandas) não preserva um índice de linha através de filter()/drop(), então
# sem isso o número de linha de uma anomalia reportada seria a posição DEPOIS
# de remover linhas de lixo/vazias, não a linha real do arquivo original.
_COL_INDICE_ORIGINAL = "_linha_original_0based"


@dataclass
class PipelineResult:
    df: pl.DataFrame
    decisions: dict[str, TypeDecision] = field(default_factory=dict)
    issues: dict[str, list[QualityIssue]] = field(default_factory=dict)
    profiles: dict[str, ColumnProfile] = field(default_factory=dict)
    structure_report: Optional[StructureReport] = None
    read_diagnostics: object = None
    verdicts: dict[str, "ve.Verdict"] = field(default_factory=dict)
    file_verdict: "ve.Verdict" = None


def process_dataframe(
    df_raw: pl.DataFrame,
    tipos_padrao_atual: Optional[dict[str, TypeDecision]] = None,
    thresholds: Optional[Thresholds] = None,
    tipos_fixos: Optional[dict[str, TypeDecision]] = None,
) -> PipelineResult:
    """Recebe um DataFrame RAW (todas as colunas string, direto do reader) e
    devolve o DataFrame final tipado + toda a auditoria do processo.

    `tipos_fixos` (schema já TRAVADO da tabela de destino, por coluna — ver
    `DatabaseManager.obter_tipos_fixos_tabela`): quando uma coluna aparece
    aqui, o tipo dela é usado TAL QUAL, sem reconciliar/alargar com o que
    este arquivo teria decidido sozinho — diferente de `tipos_padrao_atual`
    (padrão GLOBAL por nome de coluna, entre tabelas diferentes), que só
    serve de ponto de partida para colunas NOVAS ainda não fixadas nesta
    tabela. Valor que não cabe no tipo fixo vira anomalia (NULL + log, ver
    `normalization_engine.normalize_column`), nunca um ALTER TABLE — corrige
    o schema instável de antes (cada arquivo podia re-tipar/alargar a coluna
    e até tentar um ALTER COLUMN incompatível com views dependentes)."""
    thresholds = thresholds or Thresholds()
    tipos_padrao_atual = tipos_padrao_atual or {}
    tipos_fixos = tipos_fixos or {}

    # with_row_index PREPENDE a coluna (fica em 1º) — reordena pro FIM, pra
    # não interferir na checagem de linha de lixo, que olha as 3 primeiras
    # colunas REAIS por posição (ver structure_analyzer).
    colunas_originais = df_raw.columns
    df_raw = df_raw.with_row_index(name=_COL_INDICE_ORIGINAL).select([*colunas_originais, _COL_INDICE_ORIGINAL])

    novas_cols = header_cleaner.clean_and_dedupe_columns(df_raw.columns)
    df_raw = df_raw.rename(dict(zip(df_raw.columns, novas_cols)))
    nome_indice = novas_cols[-1]  # a coluna de rastreio é sempre a última após o reordenamento acima

    df_limpo, structure_report = structure_analyzer.analyze_and_clean(df_raw)

    if nome_indice in df_limpo.columns:
        indices_originais = df_limpo[nome_indice].to_list()
        df_limpo = df_limpo.drop(nome_indice)
    else:
        indices_originais = list(range(df_limpo.height))
    # a coluna de rastreio é um detalhe interno — não conta como coluna real
    # nem no relatório estrutural nem na assinatura de colunas do arquivo.
    structure_report.n_cols_raw = max(0, structure_report.n_cols_raw - 1)
    structure_report.n_cols_final -= 1 if nome_indice not in structure_report.empty_columns else 0
    if nome_indice in structure_report.empty_columns:
        structure_report.empty_columns.remove(nome_indice)

    decisions: dict[str, TypeDecision] = {}
    issues: dict[str, list[QualityIssue]] = {}
    profiles: dict[str, ColumnProfile] = {}
    colunas_tipadas: dict[str, pl.Series] = {}
    verdicts: dict[str, ve.Verdict] = {}

    # BATELADO (1/2): núcleo/sinal/forma de TODAS as colunas num único
    # .select() — chamar isso coluna a coluna é o que fazia um arquivo real
    # de ~80 colunas levar ~40s em vez de ~4s (ver ptbr_parser.py).
    intermed = ptbr_parser.batch_materialize_core_shape(df_limpo)

    tinha_conteudo_masks: dict[str, pl.Series] = {}
    conventions: dict[str, ColumnConvention] = {}
    for nome in df_limpo.columns:
        serie = df_limpo[nome]
        stripped = serie.cast(pl.Utf8, strict=False).str.strip_chars()
        tinha_conteudo_mask = serie.is_not_null() & (stripped != "")
        tinha_conteudo_masks[nome] = tinha_conteudo_mask

        shape_full = intermed[ptbr_parser._prefixo_shape(nome)]
        shapes_non_empty = shape_full.filter(tinha_conteudo_mask)

        profile = data_profiler.profile_column(serie, thresholds, shapes_full=shape_full)
        profiles[nome] = profile
        conventions[nome] = ptbr_parser.infer_column_convention_from_series(shapes_non_empty, thresholds)

    # BATELADO (2/2): string decimal canônica de TODAS as colunas, num único
    # .select(), usando a convenção já resolvida (por coluna) acima.
    normalizados = ptbr_parser.batch_materialize_normalized(intermed, df_limpo.columns, conventions)

    for nome in df_limpo.columns:
        serie = df_limpo[nome]
        profile = profiles[nome]
        tinha_conteudo_mask = tinha_conteudo_masks[nome]

        stripped = serie.cast(pl.Utf8, strict=False).str.strip_chars()
        non_empty = stripped.filter(tinha_conteudo_mask)
        shapes_non_empty = intermed[ptbr_parser._prefixo_shape(nome)].filter(tinha_conteudo_mask)
        normalized_full = normalizados[nome]
        normalized_non_empty = normalized_full.filter(tinha_conteudo_mask)

        decisao_arquivo = type_inference.infer_column_type(
            profile, non_empty, name_hint=nome, thresholds=thresholds,
            shapes_non_empty=shapes_non_empty, normalized_non_empty=normalized_non_empty,
        )
        tipo_fixo = tipos_fixos.get(nome)
        if tipo_fixo is not None:
            decisao_final = replace(tipo_fixo)
        else:
            decisao_final = type_inference.combinar_tipos(tipos_padrao_atual.get(nome), decisao_arquivo)
        # a convenção numérica (BR/ISO) é um fato deste ARQUIVO (como ele
        # formata decimal/milhar), não do padrão global nem do tipo fixo —
        # sempre usa a resolvida agora, pra normalizar corretamente os
        # valores DESTE arquivo mesmo que o tipo final já esteja travado.
        decisao_final.column_convention = decisao_arquivo.column_convention
        decisions[nome] = decisao_final

        coluna_tipada, coluna_issues = normalization_engine.normalize_column(
            serie, decisao_final, precomputed_normalized=normalized_full
        )
        for issue in coluna_issues:
            if 0 <= issue.row < len(indices_originais):
                issue.row = indices_originais[issue.row]
        colunas_tipadas[nome] = coluna_tipada
        issues[nome] = coluna_issues

        n_com_conteudo = int(tinha_conteudo_mask.sum())
        verdicts[nome] = ve.evaluate_column(n_com_conteudo, len(coluna_issues), thresholds)

    df_final = pl.DataFrame(colunas_tipadas) if colunas_tipadas else df_limpo
    file_verdict = ve.evaluate_file(verdicts)

    return PipelineResult(
        df=df_final,
        decisions=decisions,
        issues=issues,
        profiles=profiles,
        structure_report=structure_report,
        verdicts=verdicts,
        file_verdict=file_verdict,
    )


def process_file(
    caminho: str,
    tipos_padrao_atual: Optional[dict[str, TypeDecision]] = None,
    thresholds: Optional[Thresholds] = None,
    tipos_fixos: Optional[dict[str, TypeDecision]] = None,
) -> PipelineResult:
    """Lê o arquivo (CSV ou Excel, pela extensão) e roda `process_dataframe`."""
    ext = os.path.splitext(caminho)[1].lower()
    if ext in EXTENSOES_EXCEL:
        df_raw, diag = excel_reader.read_excel(caminho)
    elif ext == ".csv":
        df_raw, diag = csv_reader.read_csv(caminho)
    else:
        raise ValueError(f"Extensão não suportada: {ext}")

    if df_raw.height == 0:
        raise ValueError("Arquivo vazio (sem linhas).")

    resultado = process_dataframe(df_raw, tipos_padrao_atual, thresholds, tipos_fixos=tipos_fixos)
    resultado.read_diagnostics = diag
    return resultado
