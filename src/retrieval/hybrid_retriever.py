"""
Retrieval ibrido: combina BM25 (forte sui termini esatti/codici errore tipo
"E60", che gli embedding denso tende a diluire) con retrieval denso (forte
su similarita' semantica e parafrasi), fuso con reciprocal rank fusion pesata.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import settings
from src.audit.timing import timed
from src.ingestion.indexer import db_path, load_vectorstore
from src.ingestion.versioning import parse_version
from src.retrieval.query_processing import NormalizedQuery, normalize_query

RRF_K = 60
STALE_VERSION_FACTOR = 0.98


@dataclass
class RetrievedChunk:
    document: Document
    score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    ambiguous: bool
    candidate_models: list[str]
    off_topic: bool = False
    max_relevance_score: float = 0.0


def is_off_topic(similarity_score: float) -> bool:
    return similarity_score < settings.off_topic_similarity_threshold


def _dense_search_with_scores(query_text: str, k: int) -> list[tuple[Document, float]]:
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query_text, k=k)
    # distanza coseno: 0 = identico, 2 = opposto
    return [(doc, 1.0 - distance) for doc, distance in results]


def compute_relevance_score(raw_query: str) -> float:
    """Punteggio di rilevanza (similarita' coseno col chunk piu' vicino
    nell'indice).
    Questa funzione serve per la claibrazione della threhsold di dominio
    """
    normalized = normalize_query(raw_query)
    results = _dense_search_with_scores(normalized.corrected, k=1)
    return results[0][1] if results else 0.0

_bm25_cache: dict[tuple, BM25Retriever] = {}


def _corpus_path() -> Path:
    return settings.vectorstore_dir / "bm25_corpus.jsonl"


def _corpus_fingerprint() -> tuple | None:
    path = _corpus_path()
    if not path.exists():
        return None
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size, settings.top_k_bm25)


def _load_bm25_corpus() -> list[Document]:
    corpus_path = _corpus_path()
    docs = []
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(Document(page_content=row["text"], metadata=row["metadata"]))
    return docs


def _tokenize_bm25(testo: str) -> list[str]:
    """Tokenizzazione per BM25, al posto del default `text.split()`.

    Il default non abbassa le maiuscole e non stacca la punteggiatura, e il
    match fra query e documento e' un confronto esatto fra token: `e43` non
    trova `E43`, `l'intestazione` non trova `intestazione`, `E60` non trova
    `E60:`. Sono esattamente i termini per cui BM25 sta in questo retriever.
    """
    return re.findall(r"\w+", testo.lower())


def _get_bm25_retriever() -> BM25Retriever | None:
    fingerprint = _corpus_fingerprint()
    if fingerprint is None:
        return None
    cached = _bm25_cache.get(fingerprint)
    if cached is not None:
        return cached

    with timed("retrieval.bm25_costruzione_indice"):
        docs = _load_bm25_corpus()
        if not docs:
            return None
        retriever = BM25Retriever.from_documents(
            docs, preprocess_func=_tokenize_bm25
        )
        retriever.k = settings.top_k_bm25

    _bm25_cache.clear() # Mantiene in memoria una sola versione del corpus per volta
    _bm25_cache[fingerprint] = retriever
    return retriever


def _bm25_ranked_docs(query_text: str) -> list[Document]:
    retriever = _get_bm25_retriever()
    if retriever is None:
        return []
    return retriever.invoke(query_text)


def _reciprocal_rank_fusion(
    dense_ranked: list[Document], bm25_ranked: list[Document],
    dense_weight: float, bm25_weight: float,
) -> dict[str, float]:
    """Combina due liste ordinate (per rank, non per punteggio grezzo ) in un unico punteggio per
    chunk_id, pesato secondo BM25_WEIGHT/DENSE_WEIGHT in .env."""
    scores: dict[str, float] = {}
    for rank, doc in enumerate(dense_ranked):
        cid = doc.metadata.get("chunk_id")
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (RRF_K + rank + 1)
    for rank, doc in enumerate(bm25_ranked):
        cid = doc.metadata.get("chunk_id")
        scores[cid] = scores.get(cid, 0.0) + bm25_weight / (RRF_K + rank + 1)
    return scores


def latest_versions() -> dict[str, str]:
    """doc_id -> versione piu' recente indicizzata.

    Il confronto usa parse_version (tuple di interi) e non l'ordinamento
    fra stringhe: `max(["10.1", "2.0"])` restituirebbe "2.0".
    """
    import sqlite3

    path = db_path()
    if not path.exists():
        return {}
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT doc_id, version FROM manuals").fetchall()
    finally:
        conn.close()

    latest: dict[str, str] = {}
    for doc_id, version in rows:
        if doc_id not in latest or parse_version(version) > parse_version(latest[doc_id]):
            latest[doc_id] = version
    return latest


def _detect_ambiguity(scored: list[RetrievedChunk], user_specified_model: bool) -> tuple[bool, list[str]]:
    """Ambiguo = piu' modelli plausibili con punteggi VICINI, non solo piu'
    modelli presenti da qualche parte fra i candidati.

    La distinzione e' la differenza fra una funzione utile e una che rende il
    sistema inservibile: con piu' manuali indicizzati, "esiste almeno un
    chunk di un altro modello nei top-K" e' quasi sempre vero, e un gate cosi'
    scritto chiederebbe disambiguazione a ogni singola domanda invece di
    rispondere. Si chiede di disambiguare solo quando il secondo modello e'
    un'alternativa realmente competitiva rispetto al primo.
    """
    if user_specified_model:
        return False, []

    best_by_model: dict[str, float] = {}
    for c in scored:
        model = c.document.metadata.get("model")
        if not model:
            continue
        if model not in best_by_model or c.score > best_by_model[model]:
            best_by_model[model] = c.score

    if len(best_by_model) < 2:
        return False, sorted(best_by_model)

    ranking = sorted(best_by_model.items(), key=lambda kv: kv[1], reverse=True)
    top_score = ranking[0][1]
    if top_score <= 0:
        return False, sorted(best_by_model)

    competitive = [m for m, s in ranking if s >= top_score * settings.ambiguity_relative_margin]
    return len(competitive) > 1, sorted(competitive)


def retrieve(normalized_query: NormalizedQuery) -> RetrievalResult:
    
    with timed("retrieval.query_densa"):
        dense_results = _dense_search_with_scores(
            normalized_query.corrected, k=settings.top_k_dense
        )
    max_relevance = dense_results[0][1] if dense_results else 0.0

    if is_off_topic(max_relevance):
        return RetrievalResult(
            chunks=[], ambiguous=False, candidate_models=[],
            off_topic=True, max_relevance_score=max_relevance,
        )

    dense_docs = [doc for doc, _ in dense_results]
    with timed("retrieval.bm25"):
        bm25_docs_ranked = _bm25_ranked_docs(normalized_query.expanded)

    # Filtro per modello/firmware se specificato dall'utente
    if normalized_query.model:
        model_lower = normalized_query.model.lower()
        dense_filtered = [d for d in dense_docs if str(d.metadata.get("model", "")).lower() == model_lower]
        bm25_filtered = [d for d in bm25_docs_ranked if str(d.metadata.get("model", "")).lower() == model_lower]
        dense_docs = dense_filtered or dense_docs
        bm25_docs_ranked = bm25_filtered or bm25_docs_ranked

    with timed("retrieval.fusion_rrf"):
        fused_scores = _reciprocal_rank_fusion(
            dense_docs, bm25_docs_ranked, settings.dense_weight, settings.bm25_weight
        )
        docs_by_chunk_id = {d.metadata.get("chunk_id"): d for d in [*dense_docs, *bm25_docs_ranked]}
        
        versions = latest_versions()
        scored: list[RetrievedChunk] = []
        for chunk_id, fused_score in fused_scores.items():
            doc = docs_by_chunk_id[chunk_id]
            latest = versions.get(doc.metadata.get("doc_id"))
            is_stale = bool(latest) and parse_version(doc.metadata.get("version")) < parse_version(latest)
            factor = STALE_VERSION_FACTOR if is_stale else 1.0
            scored.append(RetrievedChunk(document=doc, score=round(fused_score * factor, 6)))

        scored.sort(key=lambda c: c.score, reverse=True)
        top = scored[: settings.top_k_final]
        ambiguous, candidate_models = _detect_ambiguity(top, bool(normalized_query.model))

    return RetrievalResult(
        chunks=top,
        ambiguous=ambiguous,
        candidate_models=candidate_models,
        off_topic=False,
        max_relevance_score=max_relevance,
    )
