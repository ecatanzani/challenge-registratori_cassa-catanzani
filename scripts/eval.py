"""
Valutazione contro l'eval set etichettato (eval/eval_set.jsonl).

DUE MODALITA'
-------------
    python scripts/eval.py           # solo retrieval. NESSUNA chiave API, gratis,
                                     # riproducibile: e' il test di regressione
                                     # da rilanciare a ogni modifica.
    python scripts/eval.py --full    # end-to-end, chiama l'LLM. Richiede la chiave
                                     # e costa: produce le metriche che dipendono
                                     # dalla generazione.

TRE CLASSI DI RIGHE, ognuna con una domanda diversa a cui rispondere:

  kind="rispondibile"             la risposta ESISTE nel manuale.
                                 -> hit@k: le pagine attese sono fra quelle recuperate?
                                 -> immagini: le figure attese vengono allegate?
                                 -> expected_fallback=false: il sistema NON deve arrendersi.

  kind="vicina_non_documentata"  argomento plausibile in un punto vendita ma
                                 assente da questo manuale (POS, lotteria scontrini,
                                 inventario: verificati uno per uno sulle 73 pagine).
                                 -> expected_fallback=true: il sistema DEVE accorgersi
                                    di non saper rispondere. E' il comportamento piu'
                                    importante da verificare in un RAG, ed e' anche
                                    l'unico che le domande palesemente fuori tema non
                                    mettono alla prova.

  kind="fuori_tema"              nessun rapporto col dominio.
                                 -> expected_fallback=true, e possibilmente intercettata
                                    dal gate senza spendere una chiamata LLM.

COME SI LEGGE IL GATE
---------------------
Non si riporta "accuratezza del gate", e la ragione e' metodologica: la soglia
e' un limite inferiore ricavato dalle sole domande in tema (vedi
docs/soglia-gate-dominio.md), quindi le due cose che ha senso misurare sono

  - FALSI POSITIVI: domande legittime bloccate. E' il vincolo, deve essere 0.
  - CHIAMATE RISPARMIATE: quante domande estranee il gate ferma gratis. E' il
    beneficio. Le altre costano una chiamata e finiscono comunque in fallback
    corretto grazie al grounding: il gate non e' il meccanismo di correttezza.

Riportare un'accuratezza complessiva mescolerebbe questi due numeri, che hanno
costi molto diversi, in una media che non significa niente.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from src.retrieval.hybrid_retriever import retrieve
from src.retrieval.query_processing import normalize_query

EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "eval_set.jsonl"


def _expected_pages(row: dict) -> list[int]:
    if "expected_pages" in row:
        return list(row["expected_pages"])
    if "expected_page" in row:
        return [row["expected_page"]]
    return []


def _load_rows() -> list[dict]:
    if not EVAL_PATH.exists():
        print(f"Eval set non trovato: {EVAL_PATH}")
        return []
    return [json.loads(l) for l in open(EVAL_PATH, "r", encoding="utf-8") if l.strip()]


def _images_available_on(doc_id: str, chunks, pages_of_interest: set[int]) -> int:
    """Quante figure pertinenti il sistema troverebbe fra i chunk recuperati,
    limitatamente alle pagine attese.

    E' la componente di RETRIEVAL della metrica immagini: dice se le figure
    giuste sono raggiungibili. Non dice se finiranno nella risposta — quello
    dipende da quali chunk l'LLM cita davvero, e si misura con --full.
    Tenerle separate serve: se questa e' 12/12 e quella di --full e' 6/12, la
    perdita e' nelle citazioni del modello, non nel retrieval.
    """
    from src.images.crop import select_relevant_image_ids

    found: set[str] = set()
    seen_pages: set[int] = set()
    for c in chunks:
        meta = c.document.metadata
        page = meta.get("page_start")
        if page not in pages_of_interest or page in seen_pages:
            continue
        seen_pages.add(page)
        try:
            image_ids = json.loads(meta.get("image_ids", "[]"))
        except (TypeError, json.JSONDecodeError):
            continue
        found.update(select_relevant_image_ids(meta.get("doc_id"), image_ids))
    return len(found)


def run_retrieval_only(rows: list[dict]) -> None:
    hits = 0
    n_answerable = 0
    latencies = []
    by_domain: dict[str, list[bool]] = defaultdict(list)
    misses: list[str] = []
    gate_by_kind: dict[str, list[bool]] = defaultdict(list)
    img_expected = img_found = 0
    img_rows = 0

    for row in rows:
        kind = row.get("kind", "rispondibile")
        t0 = time.perf_counter()
        normalized = normalize_query(row["query"])
        result = retrieve(normalized)
        latencies.append((time.perf_counter() - t0) * 1000)

        if kind != "rispondibile":
            gate_by_kind[kind].append(result.off_topic)
            esito = "FERMATA dal gate" if result.off_topic else "passa (costa 1 chiamata LLM)"
            print(f"[{kind:22}] {esito:32} score={result.max_relevance_score:.4f}  {row['query'][:48]}")
            continue

        n_answerable += 1
        retrieved = {
            (c.document.metadata.get("doc_id"), c.document.metadata.get("page_start"))
            for c in result.chunks
        }
        expected_pages = _expected_pages(row)
        expected = {(row["expected_doc_id"], p) for p in expected_pages}
        hit = bool(expected & retrieved)
        hits += hit
        by_domain[row.get("domain", "non classificato")].append(hit)
        if not hit:
            misses.append(row["query"])

        n_exp_img = row.get("expected_image_count", 0)
        img_note = ""
        if n_exp_img:
            img_rows += 1
            n_found = _images_available_on(
                row["expected_doc_id"], result.chunks, set(expected_pages)
            )
            img_expected += n_exp_img
            img_found += min(n_found, n_exp_img)
            img_note = f"  [figure attese {n_exp_img}, raggiungibili {n_found}]"

        correction_note = ""
        if normalized.corrections:
            fixed = ", ".join(f"{a!r}->{b!r}" for a, b in normalized.corrections)
            correction_note = f"  [refusi: {fixed}]"

        if result.off_topic:
            print(f"[FALSO POSITIVO !!] {row['query']!r} bloccata dal gate "
                  f"(score={result.max_relevance_score:.4f})")
            continue
        print(f"[{'HIT ' if hit else 'MISS'}] {row['query'][:52]!r:56} attese {sorted(expected_pages)}, "
              f"recuperate {sorted(p for _, p in retrieved)}{img_note}{correction_note}")

    print("\n" + "=" * 76)
    print("REPORT — modalita' retrieval (nessuna chiave API)")
    print("=" * 76)

    if n_answerable:
        print(f"\nhit@{settings.top_k_final}: {hits}/{n_answerable} ({hits / n_answerable:.1%})")

    if img_rows:
        print(f"\nImmagini — componente di retrieval")
        print(f"  Figure attese raggiungibili nei chunk recuperati: {img_found}/{img_expected} "
              f"({img_found / img_expected:.0%}) su {img_rows} domande")
        print(f"  n={img_rows} e' piccolo: in questo manuale solo 10 pagine su 73 hanno figure.")
        print(f"  Le figure allegate DAVVERO alla risposta si misurano con --full.")

    print(f"\nGate di dominio (soglia {settings.off_topic_similarity_threshold})")
    fp = sum(1 for row in rows if row.get("kind", "rispondibile") == "rispondibile") - n_answerable
    print(f"  Falsi positivi (domande legittime bloccate): 0/{n_answerable}   <-- il vincolo"
          if fp == 0 else f"  FALSI POSITIVI: {fp}   <-- VIOLAZIONE DEL VINCOLO")
    for kind, label in [("fuori_tema", "fuori tema (lontane)"),
                        ("vicina_non_documentata", "vicine, non documentate")]:
        results = gate_by_kind.get(kind, [])
        if results:
            n_stop = sum(results)
            print(f"  Chiamate LLM risparmiate su {label:24} {n_stop}/{len(results)}")
    if gate_by_kind.get("vicina_non_documentata"):
        print("  Le vicine che passano NON sono un difetto: la similarita' coseno non puo'")
        print("  distinguerle: sono gestite a valle dal grounding, che le porta a fallback.")

    if by_domain:
        print("\nCopertura per dominio (dai domini elencati nel brief):")
        for domain in sorted(by_domain):
            results = by_domain[domain]
            n_hit = sum(results)
            flag = "" if n_hit == len(results) else "   <-- da migliorare"
            print(f"  {domain:26} {n_hit}/{len(results)} ({n_hit / len(results):.0%}){flag}")

    if misses:
        print("\nDomande in tema non recuperate correttamente:")
        for q in misses:
            print(f"  - {q}")

    print(f"\nLatenza retrieval media: {sum(latencies) / len(rows):.0f} ms (budget brief: <2000 ms)")
    print("\nNon misurabili senza LLM: tasso di fallback, immagini effettivamente")
    print("allegate, latenza E2E, primo token. Usa --full.")


def run_full(rows: list[dict]) -> None:
    """End-to-end con LLM: produce le quattro metriche del brief su dati
    etichettati. Richiede ANTHROPIC_API_KEY."""
    from src.generation.chain import answer_query

    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY non configurata: --full richiede la chiave.")
        print("Senza chiave usa `python scripts/eval.py` (retrieval, gratis).")
        return

    fallback_ok = defaultdict(int)
    fallback_tot = defaultdict(int)
    img_expected = img_attached = 0
    e2e, ttft = [], []
    hits = n_answerable = 0
    errori: list[str] = []

    for row in rows:
        kind = row.get("kind", "rispondibile")
        try:
            result = answer_query(row["query"])
        except Exception as exc:
            errori.append(f"{row['query'][:40]}: {type(exc).__name__}: {exc}")
            continue

        e2e.append(result.latency_ms)
        if result.time_to_first_token_ms is not None:
            ttft.append(result.time_to_first_token_ms)

        expected_fb = row.get("expected_fallback", kind != "rispondibile")
        actual_fb = result.fallback
        fallback_tot[kind] += 1
        fallback_ok[kind] += (actual_fb == expected_fb)
        esito = "OK " if actual_fb == expected_fb else "NO "
        print(f"[{esito}] {kind:22} fallback atteso={expected_fb!s:5} ottenuto={actual_fb!s:5} "
              f"{result.latency_ms:>5}ms  {row['query'][:40]}")

        if kind != "rispondibile":
            continue
        n_answerable += 1
        pages = set(_expected_pages(row))
        if any(c.page in pages for c in result.retrieved_chunks):
            hits += 1
        n_exp = row.get("expected_image_count", 0)
        if n_exp:
            img_expected += n_exp
            img_attached += min(sum(1 for i in result.images if i.page in pages), n_exp)

    print("\n" + "=" * 76)
    print("REPORT — modalita' end-to-end (le quattro metriche del brief)")
    print("=" * 76)
    if n_answerable:
        print(f"\nhit@{settings.top_k_final}: {hits}/{n_answerable} ({hits / n_answerable:.1%})")
    if img_expected:
        print(f"\nImmagini allegate: {img_attached}/{img_expected} delle figure attese "
              f"({img_attached / img_expected:.0%})")
    print("\nTasso di fallback — correttezza della decisione, contro ground truth:")
    for kind in ("rispondibile", "vicina_non_documentata", "fuori_tema"):
        if fallback_tot.get(kind):
            print(f"  {kind:24} {fallback_ok[kind]}/{fallback_tot[kind]} "
                  f"({fallback_ok[kind] / fallback_tot[kind]:.0%})")
    tot_ok, tot = sum(fallback_ok.values()), sum(fallback_tot.values())
    if tot:
        print(f"  {'complessivo':24} {tot_ok}/{tot} ({tot_ok / tot:.0%})")
    if e2e:
        e2e.sort()
        print(f"\nLatenza E2E: mediana {e2e[len(e2e) // 2]} ms, "
              f"max {e2e[-1]} ms (budget brief: <8000 ms)")
    if ttft:
        ttft.sort()
        print(f"Primo token: mediana {ttft[len(ttft) // 2]} ms (misurato dalla domanda)")
    if errori:
        print(f"\n{len(errori)} query terminate con errore:")
        for e in errori:
            print(f"  - {e}")


def main():
    parser = argparse.ArgumentParser(description="Valutazione del RAG sull'eval set etichettato.")
    parser.add_argument("--full", action="store_true",
                        help="End-to-end con LLM (richiede ANTHROPIC_API_KEY e costa). "
                             "Senza, valuta solo il retrieval: gratis e riproducibile.")
    args = parser.parse_args()

    rows = _load_rows()
    if not rows:
        return
    run_full(rows) if args.full else run_retrieval_only(rows)


if __name__ == "__main__":
    main()
