"""Detecção de encoding para CSV, priorizando cp1252/latin1 (predominante nos
exports BR/SAP), com validação de coerência e UTF-8 como caso reconhecido.

`cp1252`/`latin1` decodificam QUASE qualquer sequência de bytes sem lançar
exceção — "tentar cp1252 primeiro e ver se funciona" não é um teste válido
(um arquivo genuinamente UTF-8 também "funcionaria", só que errado). Por
isso a ordem real de decisão é:

  1. BOM UTF-8 explícito -> utf-8-sig (autoritativo).
  2. Decodificação ESTRITA como utf-8: se funcionar e houver byte >= 0x80,
     é genuinamente UTF-8. Se funcionar e for ASCII puro, é indiferente —
     reportamos cp1252 (preferência do domínio, sem risco: ASCII é
     subconjunto de cp1252).
  3. Se utf-8 estrito falhar: decodifica como cp1252 e mede coerência
     (proporção de caracteres de substituição/controle). Coerente -> cp1252.
  4. Senão, cai pra latin1 (nunca lança exceção), com confiança baixa.

Cada leitura registra (encoding, confiança, motivo) para o relatório de
qualidade.
"""

from __future__ import annotations

from dataclasses import dataclass

_LIMIAR_COERENCIA = 0.98


@dataclass
class EncodingResult:
    encoding: str
    confidence: float
    notes: str


def coherence_score(texto: str) -> float:
    """Fração de caracteres "limpos" (sem substituição/controle inesperado).
    1.0 = perfeitamente coerente."""
    if not texto:
        return 1.0
    total = len(texto)
    ruido = texto.count("�")
    ruido += sum(1 for ch in texto if ord(ch) < 32 and ch not in "\t\r\n")
    return max(0.0, 1.0 - ruido / total)


def detect_encoding(raw_bytes: bytes, sample_size: int = 1_000_000) -> EncodingResult:
    amostra = raw_bytes[:sample_size]

    if amostra.startswith(b"\xef\xbb\xbf"):
        return EncodingResult("utf-8-sig", 1.0, "BOM UTF-8 detectado")

    try:
        texto_utf8 = amostra.decode("utf-8", errors="strict")
        tem_byte_alto = any(b >= 0x80 for b in amostra)
        if tem_byte_alto:
            return EncodingResult("utf-8", 0.95, "decodificação UTF-8 estrita bem-sucedida com bytes acentuados")
        return EncodingResult("cp1252", 0.6, "conteúdo ASCII puro (compatível com qualquer encoding); usando cp1252 (padrão do domínio BR)")
    except UnicodeDecodeError:
        pass

    texto_cp1252 = amostra.decode("cp1252", errors="replace")
    score_cp1252 = coherence_score(texto_cp1252)
    if score_cp1252 >= _LIMIAR_COERENCIA:
        return EncodingResult("cp1252", score_cp1252, f"cp1252 coerente (score={score_cp1252:.3f})")

    texto_latin1 = amostra.decode("latin1", errors="replace")
    score_latin1 = coherence_score(texto_latin1)
    return EncodingResult(
        "latin1",
        min(score_latin1, 0.5),
        f"utf-8 e cp1252 falharam/pouco coerentes (cp1252 score={score_cp1252:.3f}); fallback latin1",
    )
