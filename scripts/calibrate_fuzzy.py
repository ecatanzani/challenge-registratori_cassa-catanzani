"""Curva della soglia fuzzy usata per riconoscere i termini di dominio: refusi
ricondotti al termine giusto contro testo corretto scambiato per un termine, al
variare di FUZZY_THRESHOLD. La correzione dei refusi non passa da qui (vedi
docs/normalizzazione-query.md).
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz import fuzz

import src.retrieval.query_processing as qp
from config import settings
from src.logging.logger import setup_main_logger

SOGLIE = (70, 75, 80, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95)


def finestre_corrette() -> set[str]:
    """Finestre di 2-3 parole del testo indicizzato: italiano corretto, che non
    deve essere scambiato per un termine di dominio."""
    finestre = set()
    with open(settings.vectorstore_dir / "bm25_corpus.jsonl", encoding="utf-8") as f:
        for riga in f:
            tk = qp._TOKEN_PATTERN.findall(json.loads(riga)["text"].lower())
            for n in (2, 3):
                finestre.update(" ".join(tk[i:i + n]) for i in range(len(tk) - n + 1))
    return finestre


def refusi_sintetici(finestre: set[str], seed: int = 7) -> list[tuple[str, str]]:
    """Per ogni parola di almeno 5 lettere dei termini di dominio di 2-3 parole:
    una lettera mancante, due lettere scambiate, una lettera sostituita."""
    random.seed(seed)
    refusi = []
    for termine in qp.DOMAIN_VOCAB:
        parole = termine.split()
        if len(parole) not in (2, 3):
            continue
        for k, w in enumerate(parole):
            if len(w) < 5:
                continue
            i = random.randrange(1, len(w) - 1)
            for var in (w[:i] + w[i + 1:], w[:i] + w[i + 1] + w[i] + w[i + 2:], w[:i] + "e" + w[i + 1:]):
                r = " ".join(parole[:k] + [var] + parole[k + 1:])
                if r != termine and r not in finestre:
                    refusi.append((r, termine))
    return refusi


def riscritte(finestre: set[str]) -> set[tuple[str, str]]:
    return {(w, m) for w in finestre if (m := qp._best_vocab_match(w)) and m != w}


def main() -> None:
    logger = setup_main_logger("FuzzyCalibration")
    attuale, functional_check = qp.FUZZY_THRESHOLD, qp._promuove_parola_funzionale
    finestre = finestre_corrette()
    refusi = refusi_sintetici(finestre)
    logger.info(f"{len(finestre)} finestre di testo corretto, {len(refusi)} refusi sintetici")

    logger.info(f"{'soglia':>8} | {'refusi recuperati':>19} | {'riscritture, solo soglia':>24} | "
                f"{'riscritture, con functional_check':>24}")
    for soglia in SOGLIE:
        qp.FUZZY_THRESHOLD = soglia
        qp._promuove_parola_funzionale = functional_check
        ok = sum(qp._best_vocab_match(r) == t for r, t in refusi)
        con_functional_check = len(riscritte(finestre))
        qp._promuove_parola_funzionale = lambda w, m: False
        solo_soglia = len(riscritte(finestre))
        segno = "  <- attuale" if soglia == attuale else ""
        logger.info(f"{soglia:>8} | {ok:>7}/{len(refusi)} ({ok / len(refusi):6.1%}) | {solo_soglia:>24} | "
                    f"{con_functional_check:>24}{segno}")

    qp.FUZZY_THRESHOLD, qp._promuove_parola_funzionale = attuale, functional_check
    logger.info(f"Con la soglia attuale ({attuale}) e la functional_check:")
    for r, t in refusi:
        if qp._best_vocab_match(r) != t:
            logger.info(f"  refuso perso     {r!r} -> {t!r} (ratio {fuzz.ratio(r, t):.1f})")
    for w, m in sorted(riscritte(finestre), key=lambda x: -fuzz.ratio(*x)):
        logger.info(f"  testo riscritto  {w!r} -> {m!r} (ratio {fuzz.ratio(w, m):.1f})")


if __name__ == "__main__":
    main()
