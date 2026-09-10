"""Calibrazioen della threshold di dominio. Questa verrà usata per filtrare le domande fuori tema.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from src.retrieval.hybrid_retriever import compute_relevance_score
from src.logging.logger import setup_main_logger

EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "eval_set.jsonl"


def calcola_scarti(punteggi: list[float], prove: int = 2000, seed: int = 0) -> list[float]:
    """Di quanto cadono, sotto il minimo di un sottoinsieme di calibrazione, le
    domande legittime tenute fuori. Meta' dentro e meta' fuori a ogni prova: e'
    il rapporto che stressa di piu' la stima con pochi dati."""
    random.seed(seed)
    n = len(punteggi)
    domande_di_calibrazione = max(2, n // 2)
    scarti = []
    for _ in range(prove):
        calib = random.sample(punteggi, domande_di_calibrazione)
        soglia = min(calib)
        scarti += [soglia - s for s in punteggi if s < soglia]
    return scarti


def rounding(x: float) -> float:
    return math.floor(x * 10000) / 10000

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--interval", type=float, default=1.5,
                    help="fattore applicato allo scarto massimo osservato (default 1.5)")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    logger = setup_main_logger("ThresholdCalibration")

    elms = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8") if l.strip()]
    logger.info(f"Calcolo del punteggio di rilevanza per {len(elms)} domande...")
    per_classe: dict[str, list[float]] = {}
    for r in elms:
        per_classe.setdefault(r.get("kind", "rispondibile"), []).append(
            compute_relevance_score(r["query"]))

    # Estrazione delle domande per ogni singola classe
    con_risposta = per_classe.get("rispondibile", [])
    vicine = per_classe.get("vicina_non_documentata", [])
    lontane = per_classe.get("fuori_tema", [])
    if len(con_risposta) < 4:
        sys.exit("Servono almeno 4 domande con risposta nel manuale per stimare il margine.")

    # Calcolo degli scarti
    scarti = calcola_scarti(con_risposta)
    peggiore = max(scarti)
    margine = rounding(peggiore * args.interval)

    # Calcolo della threshold
    limiti = [(rounding(min(con_risposta) - margine), "le domande con risposta nel manuale")]
    if vicine:
        limiti.append((rounding(min(vicine)), "le vicine non documentate"))
    soglia, decide = min(limiti, key=lambda x: x[0])

    logger.info(f"OFF_TOPIC_SIMILARITY_THRESHOLD={soglia}")

    logger.info(f"\n{'=' * 74}\nREASONING\n{'=' * 74}")
    logger.info(f"Punteggi con risposta nel manuale : n={len(con_risposta)}  "
          f"min {min(con_risposta):.4f}  max {max(con_risposta):.4f}")
    if vicine:
        logger.info(f"Punteggi vicine non documentate   : n={len(vicine)}  "
              f"min {min(vicine):.4f}  max {max(vicine):.4f}")
    if lontane:
        logger.info(f"Punteggi fuori tema               : n={len(lontane)}  "
              f"min {min(lontane):.4f}  max {max(lontane):.4f}")

    logger.info(f"Scarti misurati su {len(scarti)} esclusioni: "
          f"mediano {sorted(scarti)[len(scarti) // 2]:.4f}  massimo {peggiore:.4f}")
    logger.info(f"Margine = {peggiore:.4f} x {args.interval} = {margine}")
    logger.info(f"Soglia  = min({min(con_risposta):.4f} - {margine}"
          + (f" , {min(vicine):.4f} - 0)" if vicine else ")")
          + f" = {soglia}   <- decide {decide}")

    logger.info(f"\n{'=' * 74}\nFollow-Up\n{'=' * 74}")
    legittime = con_risposta + vicine
    logger.info(f"Domande legittime bloccate : {sum(1 for s in legittime if s < soglia)}/{len(legittime)}")
    logger.info(f"Domande fuori tema fermate  : {sum(1 for s in lontane if s < soglia)}/{len(lontane)}")

if __name__ == "__main__":
    main()
