"""
Proposte di FAQ correlate per i percorsi di fallback, ricavate DALL'INDICE
invece che scritte a mano.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from config import settings

# "3.12 PROGRAMMAZIONE INTESTAZIONE SCONTRINO" -> "PROGRAMMAZIONE INTESTAZIONE SCONTRINO"
_LEADING_NUMBER = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,2}[.)]?|CAPITOLO\s+\d+|SEZIONE\s+\d+)\s*[-–:]?\s*", re.IGNORECASE)
# Titoli che non descrivono una procedura e non hanno senso come FAQ.
_NON_PROCEDURAL = re.compile(
    r"^(sommario|indice|appendice|glossario|premessa|introduzione|note|"
    r"informazioni agli utenti|descrizione generale|caratteristiche tecniche)\b",
    re.IGNORECASE,
)
MIN_TITLE_CHARS = 8
MAX_TITLE_CHARS = 70


def _clean_title(title: str) -> str | None:
    if not title:
        return None
    cleaned = _LEADING_NUMBER.sub("", title).strip(" .:-–\t")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if not (MIN_TITLE_CHARS <= len(cleaned) <= MAX_TITLE_CHARS):
        return None
    if _NON_PROCEDURAL.match(cleaned):
        return None
    if not any(ch.isalpha() for ch in cleaned):
        return None
    return cleaned


def _as_question(title: str) -> str:
    """Titolo di sezione -> domanda in linguaggio naturale.
    Crea una domanda a partire dal titolo.
    """
    acronyms = {"IVA", "DGFE", "MMC", "POS", "PC", "X", "Z", "WIFI", "RCH"}
    words = []
    for word in title.split():
        stripped = word.strip("()[]")
        words.append(word if stripped.upper() in acronyms else word.lower())
    phrase = " ".join(words)
    return f"Come si esegue: {phrase}?"


@lru_cache(maxsize=1)
def _index_sections() -> list[tuple[str, int]]:
    """(titolo_sezione_pulito, caratteri_totali) letti dal corpus BM25.

    Si legge il JSONL e non il vector store perche' qui non serve nessuna
    similarita': serve l'elenco delle sezioni, che e' un dato testuale gia'
    su disco. Nessun embedding, nessuna chiamata di rete.
    """
    corpus_path = settings.vectorstore_dir / "bm25_corpus.jsonl"
    if not corpus_path.exists():
        return []

    sizes: dict[str, int] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = _clean_title(row.get("metadata", {}).get("section_title", ""))
            if title:
                sizes[title] = sizes.get(title, 0) + len(row.get("text", ""))

    return sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)


def faqs_from_index(limit: int = 3) -> list[str]:
    """FAQ 'di orientamento': le sezioni piu' sostanziose del manuale.

    L'ordinamento e' per quantita' di testo della sezione, come proxy di
    'procedura corposa e quindi probabilmente utile'. E' un'euristica, non
    una misura di popolarita': con log d'uso reali degli operatori la si
    sostituirebbe con le sezioni effettivamente piu' consultate.
    """
    return [_as_question(title) for title, _ in _index_sections()[:limit]]


def faqs_from_chunks(chunks, limit: int = 3) -> list[str]:
    """FAQ correlate ai chunk realmente recuperati per questa domanda.

    `chunks`: list[RetrievedChunk]. Se nessuno dei chunk recuperati ha un
    titolo di sezione utilizzabile, ripiega sulle FAQ di orientamento — cosi'
    il percorso di fallback non resta mai a mani vuote.
    """
    seen: list[str] = []
    for c in chunks:
        title = _clean_title(c.document.metadata.get("section_title", ""))
        if title and title not in seen:
            seen.append(title)
        if len(seen) >= limit:
            break

    questions = [_as_question(t) for t in seen]
    if len(questions) < limit:
        for extra in faqs_from_index(limit):
            if extra not in questions:
                questions.append(extra)
            if len(questions) >= limit:
                break
    return questions[:limit]
