import os
import time
import re
import logging
import threading
from datetime import datetime
from typing import Optional
import polars as pl
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect, Text, DateTime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from livefolder import pipeline as lf_pipeline
from livefolder import postgres_writer as lf_postgres_writer
from livefolder.models import TargetType, TypeDecision

# ==========================================
# MÓDULO 1: CONFIGURAÇÕES E LOGGING
# ==========================================

import warnings

# Ignora o ruído inofensivo do openpyxl sobre planilhas sem estilo padrão
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", message="Workbook contains no default style")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

# Configuração da Base (Altere para PostgreSQL/MySQL se necessário)
# Ex: 'postgresql://postgres:senha@localhost:5432/meubanco'
DB_URI = os.getenv("DB_URI", "postgresql+psycopg2://postgres:Mandre.21162925@127.0.0.1:5432/testes")
PASTA_MONITORADA = r"C:\Users\dedex\OneDrive - Anheuser-Busch InBev\testes\GP7 - RELATÓRIOS"

# Códigos UNB (unidade de negócio) que aparecem no NOME dos arquivos (ex.:
# "02.03.04_817260.csv" -> UNB 817260), configurados aqui (.env) porque
# muitos relatórios (Grade, Produtos, etc.) não trazem esse código em
# NENHUMA coluna dos dados — só no nome do arquivo. Lista de códigos
# conhecidos, não um regex genérico: o nome do arquivo também tem outros
# números (ex.: relatório "01200127_817260.csv") que não são código de UNB.
UNB_CODIGOS_ARQUIVO = [c.strip() for c in os.getenv("UNB_CODIGOS_ARQUIVO", "").split(",") if c.strip()]


def detectar_codigo_unb_arquivo(nome_arquivo: str) -> Optional[str]:
    """Retorna o código UNB reconhecido no nome do arquivo (ver
    UNB_CODIGOS_ARQUIVO), ou None se nenhum dos códigos configurados aparecer
    — nesse caso o chamador não deve criar a coluna (arquivo sem UNB
    identificável no nome, ex.: "BASE PDVS.xlsx")."""
    base = os.path.splitext(nome_arquivo)[0]
    for codigo in UNB_CODIGOS_ARQUIVO:
        if codigo in base:
            return codigo
    return None

# ==========================================
# MÓDULO 2: BANCO DE DADOS E CONTROLE
# ==========================================
class DatabaseManager:
    def __init__(self, uri):
        self.engine = create_engine(uri)
        self._criar_tabelas_controle()

    def _criar_tabelas_controle(self):
        with self.engine.begin() as conn:
            # Adaptado para funcionar universalmente via SQLAlchemy puro
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tb_controle_arquivos (
                    id SERIAL PRIMARY KEY,
                    nome_arquivo VARCHAR(255),
                    caminho_arquivo TEXT UNIQUE,
                    tabela_destino VARCHAR(100),
                    data_modificacao_arquivo TIMESTAMP,
                    data_processamento TIMESTAMP,
                    status VARCHAR(50),
                    linhas_processadas INT,
                    linhas_ignoradas INT,
                    colunas_removidas INT,
                    tempo_processamento_segundos FLOAT,
                    mensagem_erro TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tb_log_anomalias (
                    id SERIAL PRIMARY KEY,
                    caminho_arquivo TEXT,
                    linha_arquivo INT,
                    coluna VARCHAR(100),
                    valor_problematico TEXT,
                    tipo_esperado VARCHAR(50),
                    acao VARCHAR(100),
                    motivo_rejeicao TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tb_metadados_colunas (
                    tabela VARCHAR(100),
                    coluna VARCHAR(100),
                    tipo_banco VARCHAR(50),
                    PRIMARY KEY (tabela, coluna)
                )
            """))
            # Schema FIXO por tabela: uma vez que uma coluna é gravada aqui, o
            # próximo arquivo que cair nessa MESMA tabela usa esse tipo TAL
            # QUAL (ver DatabaseManager.obter_tipos_fixos_tabela) — não
            # re-infere/alarga a cada arquivo. Elimina o schema instável de
            # antes: cada arquivo podia mudar o tipo de uma coluna (inclusive
            # tentando um ALTER COLUMN incompatível com view dependente) só
            # por causa da própria amostra. Para corrigir um tipo errado, o
            # usuário edita a linha (tabela, coluna) aqui — a tabela real
            # evolui sozinha pra bater com o que está aqui (ver
            # `coluna_precisa_evoluir` em conversor_db.py).
            for coluna_extra in (
                "escala INT", "precisao INT", "varchar_len INT",
                "is_identificador BOOLEAN", "tipo_identificador VARCHAR(20)",
            ):
                conn.execute(text(f"ALTER TABLE tb_metadados_colunas ADD COLUMN IF NOT EXISTS {coluna_extra}"))
            # Registro GLOBAL (por nome de coluna, não por tabela): garante que a
            # mesma coluna (ex: 'codigo_material') receba sempre o MESMO tipo em
            # todos os arquivos/tabelas, mesmo quando processados em ordens/dias
            # diferentes. Sem isso, cada arquivo é tipado isoladamente e a mesma
            # coluna pode virar INTEGER numa tabela e TEXT em outra.
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tb_tipos_colunas_padrao (
                    coluna VARCHAR(100) PRIMARY KEY,
                    tipo_banco VARCHAR(50) NOT NULL,
                    ultima_tabela_origem VARCHAR(100),
                    data_atualizacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            # Colunas adicionadas pelo motor LiveFolder novo (escala/precisão de
            # NUMERIC, largura de VARCHAR, identificador CPF/CNPJ/EAN) — em cima
            # da tabela que já existia, via ADD COLUMN IF NOT EXISTS idempotente,
            # pra não perder o padrão global já consolidado em bancos existentes.
            for coluna_extra in (
                "escala INT", "precisao INT", "varchar_len INT",
                "is_identificador BOOLEAN", "tipo_identificador VARCHAR(20)",
            ):
                conn.execute(text(f"ALTER TABLE tb_tipos_colunas_padrao ADD COLUMN IF NOT EXISTS {coluna_extra}"))

    def arquivo_ja_processado(self, caminho: str, mtime: float) -> bool:
        """Verifica se o arquivo já foi processado com sucesso na mesma versão."""
        query = text("""
            SELECT 1 FROM tb_controle_arquivos 
            WHERE caminho_arquivo = :caminho 
            AND data_modificacao_arquivo = :mtime 
            AND status = 'SUCESSO'
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"caminho": caminho, "mtime": datetime.fromtimestamp(mtime)}).fetchone()
            return result is not None

    def registrar_inicio(self, caminho: str) -> None:
        """Reseta status do arquivo (e anomalias antigas) para nova tentativa."""
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM tb_controle_arquivos WHERE caminho_arquivo = :caminho"), {"caminho": caminho})
            conn.execute(text("DELETE FROM tb_log_anomalias WHERE caminho_arquivo = :caminho"), {"caminho": caminho})

    def registrar_anomalias(self, conn, caminho_arquivo: str, anomalias: list) -> None:
        """Grava, um registro por valor problemático, os dados que não puderam ser
        convertidos para o tipo definido da coluna (viraram NULL na carga) — em vez
        de deixar a coluna inteira cair para TEXT por causa de poucas linhas sujas."""
        if not anomalias:
            return
        query = text("""
            INSERT INTO tb_log_anomalias
            (caminho_arquivo, linha_arquivo, coluna, valor_problematico, tipo_esperado, acao, motivo_rejeicao)
            VALUES (:caminho, :linha, :coluna, :valor, :tipo, 'Convertido para NULL', :motivo)
        """)
        for a in anomalias:
            valor = a["valor"]
            conn.execute(query, {
                "caminho": caminho_arquivo,
                "linha": a["linha"],
                "coluna": a["coluna"],
                "valor": valor[:2000] if isinstance(valor, str) else valor,
                "tipo": a["tipo_esperado"],
                "motivo": a["motivo"],
            })

    def atualizar_metadados_colunas(self, conn, tabela: str, decisions: dict) -> None:
        """Grava/atualiza o schema FIXO (tipo completo, com escala/precisão/
        largura) de cada coluna da tabela — fonte de verdade lida de volta em
        `obter_tipos_fixos_tabela` no próximo arquivo desta mesma tabela."""
        query = text("""
            INSERT INTO tb_metadados_colunas
                (tabela, coluna, tipo_banco, escala, precisao, varchar_len,
                 is_identificador, tipo_identificador)
            VALUES (:tabela, :coluna, :tipo, :escala, :precisao, :varchar_len,
                    :is_id, :tipo_id)
            ON CONFLICT (tabela, coluna) DO UPDATE SET
                tipo_banco = EXCLUDED.tipo_banco,
                escala = EXCLUDED.escala,
                precisao = EXCLUDED.precisao,
                varchar_len = EXCLUDED.varchar_len,
                is_identificador = EXCLUDED.is_identificador,
                tipo_identificador = EXCLUDED.tipo_identificador
        """)
        for coluna, decisao in decisions.items():
            conn.execute(query, {
                "tabela": tabela, "coluna": coluna, "tipo": decisao.target_type.value,
                "escala": decisao.scale, "precisao": decisao.precision, "varchar_len": decisao.varchar_len,
                "is_id": decisao.is_identifier, "tipo_id": decisao.identifier_kind,
            })

    def obter_tipos_fixos_tabela(self, tabela: str) -> dict:
        """Carrega o schema FIXO já travado (coluna -> TypeDecision) desta
        tabela específica — usado para que o próximo arquivo NÃO re-infira/
        alargue o tipo de uma coluna já estabelecida (ver `tipos_fixos` em
        `livefolder.pipeline`). Diferente de `obter_tipos_padrao_colunas`
        (padrão GLOBAL por nome de coluna, entre tabelas diferentes)."""
        query = text("""
            SELECT coluna, tipo_banco, escala, precisao, varchar_len,
                   is_identificador, tipo_identificador
            FROM tb_metadados_colunas
            WHERE tabela = :tabela
        """)
        with self.engine.connect() as conn:
            resultado = {}
            for row in conn.execute(query, {"tabela": tabela}):
                resultado[row.coluna] = TypeDecision(
                    target_type=TargetType(row.tipo_banco),
                    scale=row.escala,
                    precision=row.precisao,
                    varchar_len=row.varchar_len,
                    is_identifier=bool(row.is_identificador),
                    identifier_kind=row.tipo_identificador,
                    confidence=1.0,
                    reason="tipo fixo da tabela",
                )
            return resultado

    def obter_tipos_padrao_colunas(self) -> dict:
        """Carrega o padrão global (coluna -> TypeDecision) já consolidado a
        partir de todos os arquivos já processados até agora."""
        query = text("""
            SELECT coluna, tipo_banco, escala, precisao, varchar_len,
                   is_identificador, tipo_identificador
            FROM tb_tipos_colunas_padrao
        """)
        with self.engine.connect() as conn:
            resultado = {}
            for row in conn.execute(query):
                resultado[row.coluna] = TypeDecision(
                    target_type=TargetType(row.tipo_banco),
                    scale=row.escala,
                    precision=row.precisao,
                    varchar_len=row.varchar_len,
                    is_identifier=bool(row.is_identificador),
                    identifier_kind=row.tipo_identificador,
                    confidence=1.0,
                    reason="padrão global consolidado",
                )
            return resultado

    def atualizar_tipos_padrao_colunas(self, conn, tabela: str, decisions: dict) -> None:
        """Grava o tipo final (já conciliado com o padrão global) de cada coluna.
        Como a conciliação em type_inference.combinar_tipos só "anda" em direção
        ao tipo mais abrangente (nunca de TEXT de volta para numérico), este
        registro fica cada vez mais estável a cada novo arquivo processado."""
        query = text("""
            INSERT INTO tb_tipos_colunas_padrao
                (coluna, tipo_banco, escala, precisao, varchar_len, is_identificador,
                 tipo_identificador, ultima_tabela_origem, data_atualizacao)
            VALUES (:coluna, :tipo, :escala, :precisao, :varchar_len, :is_id,
                    :tipo_id, :tabela, CURRENT_TIMESTAMP)
            ON CONFLICT (coluna) DO UPDATE SET
                tipo_banco = EXCLUDED.tipo_banco,
                escala = EXCLUDED.escala,
                precisao = EXCLUDED.precisao,
                varchar_len = EXCLUDED.varchar_len,
                is_identificador = EXCLUDED.is_identificador,
                tipo_identificador = EXCLUDED.tipo_identificador,
                ultima_tabela_origem = EXCLUDED.ultima_tabela_origem,
                data_atualizacao = CURRENT_TIMESTAMP
        """)
        for coluna, decisao in decisions.items():
            conn.execute(query, {
                "coluna": coluna, "tipo": decisao.target_type.value, "escala": decisao.scale,
                "precisao": decisao.precision, "varchar_len": decisao.varchar_len,
                "is_id": decisao.is_identifier, "tipo_id": decisao.identifier_kind, "tabela": tabela,
            })

EXTENSOES_SUPORTADAS = (".csv", ".xlsx", ".xls", ".xlsm", ".xlsb")

# ==========================================
# MÓDULO 4: INFERÊNCIA DE TIPOS
# ==========================================
# A tipagem (leitura, encoding, delimitador, limpeza estrutural, perfilamento,
# parser numérico PT-BR, inferência de tipo, normalização) mora inteira no
# pacote `livefolder/` — ver `livefolder/pipeline.py` (ponto de integração
# único) e os módulos que ele orquestra (`ptbr_parser`, `type_inference`,
# `data_profiler`, `normalization_engine`, `readers/*`).

# ==========================================
# MÓDULO 5: ORQUESTRAÇÃO E CARGA LOTE
# ==========================================
class IngestionEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        # Serializa o processamento do MESMO arquivo. A varredura inicial roda na
        # thread principal enquanto o Watchdog dispara eventos numa thread
        # separada — os dois podem chamar processar_arquivo para o mesmo caminho
        # ao mesmo tempo. Sem trava, ambos "processam com SUCESSO" e o segundo a
        # gravar estoura UniqueViolation em tb_controle_arquivos (e ainda faz
        # DELETE/append concorrente na tabela de destino).
        self._em_processamento_lock = threading.Lock()
        self._em_processamento = set()

    def processar_arquivo(self, caminho: str):
        caminho_norm = os.path.normcase(os.path.abspath(caminho))
        with self._em_processamento_lock:
            if caminho_norm in self._em_processamento:
                log.info(f"⏭️ IGNORADO (já em processamento): {os.path.basename(caminho)}")
                return
            self._em_processamento.add(caminho_norm)
        try:
            self._processar_arquivo(caminho)
        finally:
            with self._em_processamento_lock:
                self._em_processamento.discard(caminho_norm)

    def _processar_arquivo(self, caminho: str):
        t0 = time.time()
        nome_arquivo = os.path.basename(caminho)
        # Tabela nomeada pela PASTA que contém o arquivo (como antes), não pelo
        # nome do próprio arquivo: arquivos recorrentes de um mesmo relatório
        # (ex: "d_02.02.53\01-2024_817260.csv", "d_02.02.53\02-2024_817260.csv")
        # ficam todos na pasta "d_02.02.53" e devem cair na mesma tabela.
        nome_pasta = os.path.basename(os.path.dirname(caminho)) or os.path.splitext(nome_arquivo)[0]
        tabela_destino = f"t_{nome_pasta.lower()}"
        tabela_destino = re.sub(r'[^\w]', '_', tabela_destino)
        mtime = os.path.getmtime(caminho)

        # 1. Validação Incremental
        if self.db.arquivo_ja_processado(caminho, mtime):
            log.info(f"⏭️ IGNORADO (Não modificado): {nome_arquivo}")
            return

        log.info(f"⚙️ PROCESSANDO: {nome_arquivo}")
        self.db.registrar_inicio(caminho)

        try:
            # 2-4. Leitura + limpeza estrutural + tipagem — tudo em
            # livefolder.pipeline. `tipos_fixos_tabela` é o schema já
            # TRAVADO desta tabela (ver obter_tipos_fixos_tabela): colunas
            # já estabelecidas usam esse tipo tal qual, sem re-inferir a
            # cada arquivo — só coluna NOVA nesta tabela usa o padrão
            # GLOBAL por nome (`tipos_padrao_atual`) como ponto de partida.
            tipos_padrao_atual = self.db.obter_tipos_padrao_colunas()
            tipos_fixos_tabela = self.db.obter_tipos_fixos_tabela(tabela_destino)
            resultado = lf_pipeline.process_file(
                caminho, tipos_padrao_atual=tipos_padrao_atual, tipos_fixos=tipos_fixos_tabela,
            )

            if resultado.df.height == 0:
                raise ValueError("Arquivo ficou vazio após a limpeza de dados.")

            for coluna, decisao_final in resultado.decisions.items():
                if coluna in tipos_fixos_tabela:
                    continue  # tipo já travado desta tabela — nada novo a logar
                tipo_padrao_anterior = tipos_padrao_atual.get(coluna)
                if tipo_padrao_anterior and tipo_padrao_anterior.target_type != decisao_final.target_type:
                    log.info(
                        f"↔️ Coluna '{coluna}' padronizada como {decisao_final.target_type.value} "
                        f"para manter o mesmo tipo em todas as tabelas."
                    )

            # Rastreia de qual arquivo cada linha veio, permitindo reprocessar um
            # arquivo modificado sem duplicar linhas (delete-by-origem antes do append).
            # + código UNB derivado do NOME do arquivo, só quando reconhecido (ver
            # detectar_codigo_unb_arquivo) — muitos relatórios não trazem esse
            # código em nenhuma coluna dos dados. + timestamp de modificação do
            # arquivo, para saber de QUANDO é a informação de cada linha (não só
            # de quando ela foi carregada no banco).
            colunas_extra = [pl.lit(nome_arquivo).alias("arquivo_origem")]
            codigo_unb_arquivo = detectar_codigo_unb_arquivo(nome_arquivo)
            if codigo_unb_arquivo:
                colunas_extra.append(pl.lit(codigo_unb_arquivo).alias("unb_arquivo"))
            colunas_extra.append(pl.lit(datetime.fromtimestamp(mtime)).alias("arquivo_atualizado_em"))
            df_final = resultado.df.with_columns(colunas_extra)

            # 5. Carga em Lote (Salvando de forma eficiente, com tipos de coluna explícitos)
            dtype_sql = {col: lf_postgres_writer.sqlalchemy_type_for(dec) for col, dec in resultado.decisions.items()}
            if codigo_unb_arquivo:
                dtype_sql["unb_arquivo"] = Text
            dtype_sql["arquivo_atualizado_em"] = DateTime
            df_pandas = lf_postgres_writer.to_pandas_for_insert(df_final, resultado.decisions)

            # Anomalias: valores que não bateram com o tipo decidido da coluna
            # (viraram NULL na carga) — uma por linha, para auditoria em
            # tb_log_anomalias, em vez de desaparecerem em silêncio.
            anomalias_arquivo = [
                {
                    "linha": issue.row + 2,  # +1 header, +1 índice 0-based -> nº da linha na planilha
                    "coluna": issue.column,
                    "valor": issue.raw_value,
                    "tipo_esperado": resultado.decisions[coluna].target_type.value,
                    "motivo": issue.reason,
                }
                for coluna, issues in resultado.issues.items()
                for issue in issues
            ]

            with self.db.engine.begin() as conn:
                tabela_existe = inspect(conn).has_table(tabela_destino)

                if tabela_existe:
                    # Tabela compartilhada por todos os arquivos da mesma pasta —
                    # sempre MERGE (append), nunca 'replace' (que apagaria os
                    # dados dos outros arquivos já carregados nesta mesma
                    # tabela). Coluna já fixada (ver tipos_fixos_tabela acima)
                    # não muda de tipo sozinha aqui: `decisao` já É o tipo fixo,
                    # então `coluna_precisa_evoluir` só dispara em duas
                    # situações — coluna nova nesta tabela (schema ainda não
                    # tinha essa coluna) ou o USUÁRIO editou manualmente o tipo
                    # fixo em tb_metadados_colunas (aí o ALTER sincroniza a
                    # tabela real com o novo padrão).
                    colunas_info = {c["name"]: c["type"] for c in inspect(conn).get_columns(tabela_destino)}
                    for col, decisao in resultado.decisions.items():
                        tipo_sql_nome = lf_postgres_writer.sql_type_name_for(decisao)
                        if col not in colunas_info:
                            conn.execute(text(f'ALTER TABLE "{tabela_destino}" ADD COLUMN "{col}" {tipo_sql_nome}'))
                            log.info(f"➕ Coluna nova '{col}' ({tipo_sql_nome}) adicionada à tabela existente '{tabela_destino}'.")
                        else:
                            tipo_refletido = colunas_info[col]
                            if lf_postgres_writer.coluna_precisa_evoluir(tipo_refletido, decisao):
                                conn.execute(text(
                                    f'ALTER TABLE "{tabela_destino}" ALTER COLUMN "{col}" '
                                    f'TYPE {tipo_sql_nome} USING "{col}"::{tipo_sql_nome}'
                                ))
                                log.info(
                                    f"🔧 Coluna '{col}' evoluída de {tipo_refletido} para {tipo_sql_nome} "
                                    f"na tabela '{tabela_destino}'."
                                )

                    # Colunas derivadas (fora da tipagem do arquivo, ver acima): podem
                    # não existir ainda se o primeiro arquivo deste grupo não tinha um
                    # código UNB reconhecível no nome — adiciona sob demanda.
                    if codigo_unb_arquivo and "unb_arquivo" not in colunas_info:
                        conn.execute(text(f'ALTER TABLE "{tabela_destino}" ADD COLUMN "unb_arquivo" TEXT'))
                        log.info(f"➕ Coluna nova 'unb_arquivo' (TEXT) adicionada à tabela existente '{tabela_destino}'.")
                    if "arquivo_atualizado_em" not in colunas_info:
                        conn.execute(text(f'ALTER TABLE "{tabela_destino}" ADD COLUMN "arquivo_atualizado_em" TIMESTAMP'))
                        log.info(f"➕ Coluna nova 'arquivo_atualizado_em' (TIMESTAMP) adicionada à tabela existente '{tabela_destino}'.")

                    # Idempotência: remove a versão anterior deste MESMO arquivo antes
                    # de reinserir, para que reprocessar um arquivo modificado não duplique linhas.
                    conn.execute(
                        text(f'DELETE FROM "{tabela_destino}" WHERE arquivo_origem = :nome'),
                        {"nome": nome_arquivo},
                    )
                    df_pandas.to_sql(tabela_destino, con=conn, if_exists='append', index=False, dtype=dtype_sql)
                else:
                    df_pandas.to_sql(tabela_destino, con=conn, if_exists='replace', index=False, dtype=dtype_sql)

                # Registra o tipo de cada coluna para consulta futura
                self.db.atualizar_metadados_colunas(conn, tabela_destino, resultado.decisions)
                # Atualiza o padrão global (por nome de coluna) usado na próxima carga
                self.db.atualizar_tipos_padrao_colunas(conn, tabela_destino, resultado.decisions)

                # Registra os valores que não bateram com o tipo da coluna (viraram
                # NULL) para auditoria, em vez de deixá-los desaparecer em silêncio.
                if anomalias_arquivo:
                    self.db.registrar_anomalias(conn, caminho, anomalias_arquivo)
                    log.warning(
                        f"⚠️ {len(anomalias_arquivo)} valor(es) não reconhecido(s) no formato "
                        f"esperado viraram NULL — detalhes em tb_log_anomalias."
                    )

                # Registra sucesso no controle
                tempo_exec = time.time() - t0
                sr = resultado.structure_report
                conn.execute(text("""
                    INSERT INTO tb_controle_arquivos
                    (nome_arquivo, caminho_arquivo, tabela_destino, data_modificacao_arquivo, data_processamento, status, linhas_processadas, linhas_ignoradas, colunas_removidas, tempo_processamento_segundos)
                    VALUES (:nome, :caminho, :tabela, :mtime, CURRENT_TIMESTAMP, 'SUCESSO', :linhas, :ignoradas, :col_rem, :tempo)
                    ON CONFLICT (caminho_arquivo) DO UPDATE SET
                        nome_arquivo = EXCLUDED.nome_arquivo,
                        tabela_destino = EXCLUDED.tabela_destino,
                        data_modificacao_arquivo = EXCLUDED.data_modificacao_arquivo,
                        data_processamento = EXCLUDED.data_processamento,
                        status = EXCLUDED.status,
                        linhas_processadas = EXCLUDED.linhas_processadas,
                        linhas_ignoradas = EXCLUDED.linhas_ignoradas,
                        colunas_removidas = EXCLUDED.colunas_removidas,
                        tempo_processamento_segundos = EXCLUDED.tempo_processamento_segundos,
                        mensagem_erro = NULL
                """), {
                    "nome": nome_arquivo, "caminho": caminho, "tabela": tabela_destino,
                    "mtime": datetime.fromtimestamp(mtime), "linhas": resultado.df.height,
                    "ignoradas": sr.junk_rows_dropped + sr.fully_empty_rows_dropped,
                    "col_rem": len(sr.empty_columns), "tempo": round(tempo_exec, 2)
                })

            log.info(
                f"✅ SUCESSO: {nome_arquivo} | {resultado.df.height} linhas em {tempo_exec:.2f}s "
                f"| status={resultado.file_verdict.value}"
            )

        except Exception as e:
            # 6. Log Estruturado de Erro. Nunca deixa a gravação do erro derrubar
            # o sistema: se o próprio INSERT de erro falhar (ex.: corrida com
            # outra thread que já registrou este caminho), apenas loga e segue.
            log.error(f"❌ ERRO em {nome_arquivo}: {str(e)}")
            try:
                with self.db.engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO tb_controle_arquivos
                        (nome_arquivo, caminho_arquivo, tabela_destino, data_modificacao_arquivo, data_processamento, status, mensagem_erro)
                        VALUES (:nome, :caminho, :tabela, :mtime, CURRENT_TIMESTAMP, 'ERRO', :erro)
                        ON CONFLICT (caminho_arquivo) DO UPDATE SET
                            nome_arquivo = EXCLUDED.nome_arquivo,
                            tabela_destino = EXCLUDED.tabela_destino,
                            data_modificacao_arquivo = EXCLUDED.data_modificacao_arquivo,
                            data_processamento = EXCLUDED.data_processamento,
                            status = EXCLUDED.status,
                            mensagem_erro = EXCLUDED.mensagem_erro
                    """), {
                        "nome": nome_arquivo, "caminho": caminho, "tabela": tabela_destino,
                        "mtime": datetime.fromtimestamp(mtime), "erro": str(e)
                    })
            except Exception as reg_err:
                log.error(f"⚠️ Falha ao registrar erro de {nome_arquivo} em tb_controle_arquivos: {reg_err}")

# ==========================================
# MÓDULO 6: MONITORAMENTO DE DIRETÓRIOS
# ==========================================
class Watcher(FileSystemEventHandler):
    def __init__(self, engine: IngestionEngine):
        self.engine = engine

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(EXTENSOES_SUPORTADAS):
            # Pequeno delay para garantir que o arquivo terminou de ser copiado/salvo
            time.sleep(1)
            self.engine.processar_arquivo(event.src_path)

    def on_created(self, event):
        self.on_modified(event)

def iniciar_sistema():
    os.makedirs(PASTA_MONITORADA, exist_ok=True)
    db = DatabaseManager(DB_URI)
    engine = IngestionEngine(db)

    # Inicia o Watchdog ANTES da varredura inicial — a varredura pode levar
    # minutos numa pasta com centenas de arquivos; se o Watchdog só começar
    # DEPOIS dela, qualquer modificação feita nesse meio-tempo só é notada
    # quando a varredura terminar (achado real: "demora muito pra perceber
    # que um arquivo mudou"). Começando antes, a modificação é capturada na
    # hora, mesmo com a varredura inicial ainda rodando — processar o mesmo
    # arquivo duas vezes (evento ao vivo + varredura) é seguro/idempotente
    # (ver arquivo_ja_processado / delete-by-origem antes do append).
    log.info(f"👁️ Monitorando alterações na pasta: {PASTA_MONITORADA} ...")
    observer = Observer()
    observer.schedule(Watcher(engine), PASTA_MONITORADA, recursive=True)
    observer.start()

    # Carga inicial da pasta
    log.info("🔍 Iniciando varredura inicial...")
    for root, _, files in os.walk(PASTA_MONITORADA):
        for file in files:
            if file.lower().endswith(EXTENSOES_SUPORTADAS):
                engine.processar_arquivo(os.path.join(root, file))
    log.info("✅ Varredura inicial concluída.")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
        log.info("🛑 Sistema encerrado.")
    observer.join()

if __name__ == "__main__":
    iniciar_sistema()