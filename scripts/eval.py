"""Valuta il sistema completo sull'eval set etichettato.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from src.logging.logger import setup_main_logger
from src.generation.chain import answer_query

EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "eval_set.jsonl"
CLASSI = ("rispondibile", "vicina_non_documentata", "fuori_tema")

def main() -> None:
    logger = setup_main_logger("eval")
    if not settings.anthropic_api_key:
        sys.exit("ANTHROPIC_API_KEY non configurata: la valutazione end-to-end richiede la chiave.")
    if not EVAL_PATH.exists():
        sys.exit(f"Eval set non trovato: {EVAL_PATH}")
    rows = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8") if l.strip()]

    decisioni_ok: dict[str, int] = defaultdict(int)
    decisioni_tot: dict[str, int] = defaultdict(int)
    figure_attese = figure_allegate = 0
    hit = rispondibili = 0
    e2e: list[int] = []
    ttft: list[int] = []
    errs: list[str] = []

    for row in rows:
        kind = row["kind"]
        try:
            risposta = answer_query(row["query"])
        except Exception as exc:
            errs.append(f"{row['query'][:40]}: {type(exc).__name__}: {exc}")
            continue

        e2e.append(risposta.latency_ms)
        if risposta.time_to_first_token_ms is not None:
            ttft.append(risposta.time_to_first_token_ms)

        fallback_atteso = row["expected_fallback"]
        corretta = risposta.fallback == fallback_atteso
        decisioni_tot[kind] += 1
        decisioni_ok[kind] += corretta
        
        if kind != "rispondibile":
            continue
        
        rispondibili += 1
        pagine = set(row["expected_pages"])
        hit += any(c.page in pagine for c in risposta.retrieved_chunks)
        attese = row["expected_image_count"]
        figure_attese += attese
        figure_allegate += min(sum(1 for i in risposta.images if i.page in pagine), attese)

    logger.info("\n" + "=" * 76 + "\nREPORT\n" + "=" * 76)
    if rispondibili:
        logger.info(f"\nhit@{settings.top_k_final}: {hit}/{rispondibili} ({hit / rispondibili:.1%})")
    if figure_attese:
        logger.info(f"Immagini allegate: {figure_allegate}/{figure_attese} delle figure attese "
              f"({figure_allegate / figure_attese:.0%})")

    logger.info("\nDecisione di rispondere o arrendersi, contro ground truth:")
    for kind in CLASSI:
        if decisioni_tot[kind]:
            logger.info(f"  {kind:24} {decisioni_ok[kind]}/{decisioni_tot[kind]} "
                  f"({decisioni_ok[kind] / decisioni_tot[kind]:.0%})")
    ok, tot = sum(decisioni_ok.values()), sum(decisioni_tot.values())
    if tot:
        logger.info(f"  {'complessivo':24} {ok}/{tot} ({ok / tot:.0%})")

    if e2e:
        e2e.sort()
        logger.info(f"\nLatenza E2E: mediana {e2e[len(e2e) // 2]} ms, max {e2e[-1]} ms "
              f"(budget: <8000 ms)")
    if ttft:
        ttft.sort()
        logger.info(f"Primo token: mediana {ttft[len(ttft) // 2]} ms (misurato dalla domanda)")
    if errs:
        logger.info(f"\n{len(errs)} query terminate con errore:")
        for e in errs:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
