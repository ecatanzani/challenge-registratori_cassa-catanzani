"""
Re-ranking opzionale con cross-encoder multilingue: la fusion BM25+dense da'
un buon recall ma un ranking approssimativo (proxy sul rank, non uno score
di rilevanza vero). Il cross-encoder guarda (query, chunk) insieme e da' un
punteggio di rilevanza molto piu' preciso per il top-N, a costo di qualche
decina di ms extra: lo applichiamo solo sui top_k_final*2 candidati, non su
tutto il corpus, per restare dentro il budget di latenza (TTFT < 2s).
"""
from functools import lru_cache

from config import settings
from src.audit.timing import timed
from src.retrieval.hybrid_retriever import RetrievedChunk

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    with timed("rerank.caricamento_modello"):
        return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not settings.rerank_enabled or not chunks:
        return chunks
    try:
        model = _get_cross_encoder()
    except Exception:
        return chunks

    with timed(f"rerank.predict ({len(chunks)} candidati)"):
        pairs = [(query, c.document.page_content) for c in chunks]
        scores = model.predict(pairs)
        reranked = [
            RetrievedChunk(document=c.document, score=float(s)) for c, s in zip(chunks, scores)
        ]
        reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked
