"""
Audit log locale (JSONL, append-only) per tracciabilita': query, chunk_id
consultati, versioni documento, esito (fallback si'/no), latenza. Requisito
funzionale esplicito del brief ("Log di audit ... per tracciabilita'").

Include anche compute_metrics() per calcolare le metriche osservabili
richieste nella sezione "Performance & Operativita'" del brief: latency,
hit@k (qui interpretata come "almeno un chunk sopra soglia" come proxy,
raffinabile con un eval set etichettato, vedi scripts/eval.py), tasso di
immagini allegate, tasso di fallback.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from config import settings
from src.generation.schemas import FinalAnswer


def log_query(raw_query: str, answer: FinalAnswer) -> None:
    settings.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": raw_query,
        "query_corrected": answer.understanding.corrected if answer.understanding else None,
        "corrections": answer.understanding.corrections if answer.understanding else [],
        "matched_terms": answer.understanding.matched_terms if answer.understanding else [],
        "fallback": answer.fallback,
        "off_topic": answer.off_topic,
        "ambiguous_models": answer.ambiguous_models,
        "n_steps": len(answer.steps),
        "n_images": len(answer.images),
        "retrieved_chunk_ids": [c.chunk_id for c in answer.retrieved_chunks],
        "retrieved_docs": sorted({c.doc_id for c in answer.retrieved_chunks}),
        "retrieved_versions": sorted({c.version for c in answer.retrieved_chunks if c.version}),
        "ocr_sourced_citations": sum(1 for c in answer.citations if c.ocr_used),
        "time_to_first_token_ms": answer.time_to_first_token_ms,
        "retrieval_ms": answer.retrieval_ms,
        "latency_ms": answer.latency_ms,
        "request_id": answer.request_id,
    }
    with open(settings.audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_metrics() -> dict:
    if not settings.audit_log_path.exists():
        return {"n_queries": 0}

    records = []
    with open(settings.audit_log_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        return {"n_queries": 0}

    n = len(records)
    latencies = sorted(r["latency_ms"] for r in records)
    retrieval_times = sorted(r.get("retrieval_ms", 0) for r in records)
    ttft_values = sorted(r["time_to_first_token_ms"] for r in records if r.get("time_to_first_token_ms") is not None)
    fallback_rate = sum(1 for r in records if r["fallback"]) / n
    off_topic_rate = sum(1 for r in records if r.get("off_topic")) / n
    images_rate = sum(1 for r in records if r["n_images"] > 0) / n
    hit_rate = sum(1 for r in records if r["retrieved_chunk_ids"]) / n
    corrected_rate = sum(1 for r in records if r.get("corrections")) / n

    answered = [r for r in records if not r.get("off_topic")]
    retrieval_within = (
        sum(1 for r in answered if r.get("retrieval_ms", 0) < 2000) / len(answered)
        if answered else 0.0
    )
    e2e_within = (
        sum(1 for r in answered if r["latency_ms"] < 8000) / len(answered)
        if answered else 0.0
    )

    def percentile(data, p):
        if not data:
            return 0
        idx = min(int(len(data) * p), len(data) - 1)
        return data[idx]

    return {
        "n_queries": n,
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "retrieval_ms_p50": percentile(retrieval_times, 0.5),
        "retrieval_ms_p95": percentile(retrieval_times, 0.95),
        "ttft_ms_p50": percentile(ttft_values, 0.5),
        "ttft_ms_p95": percentile(ttft_values, 0.95),
        "ttft_measured_rate": round(len(ttft_values) / n, 3),
        "retrieval_within_budget_rate": round(retrieval_within, 3),
        "e2e_within_budget_rate": round(e2e_within, 3),
        "fallback_rate": round(fallback_rate, 3),
        "off_topic_rate": round(off_topic_rate, 3),
        "images_attached_rate": round(images_rate, 3),
        "hit_rate_any_chunk": round(hit_rate, 3),
        "typo_corrected_rate": round(corrected_rate, 3),
    }
