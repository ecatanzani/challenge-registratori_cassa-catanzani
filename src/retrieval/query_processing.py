"""
Normalizzazione della query utente prima del retrieval. Produce:

- `original`  : domanda originale dell'utente.
- `corrected` : domanda con i refusi corretti sul vocabolario
                di dominio ("strono scontrino" -> "storno scontrino"). La forma della
                domanda è la medesima e verrà utilizzata dal retriever denso
- `expanded`  : domanda corretta piu' tutte le varianti sinonimiche note
                ("annullo scontrino | storno scontrino | ...") per il retriever BM25
"""

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

# Sinonimi di dominio
SYNONYMS: dict[str, list[str]] = {
    # --- annullo / storno ---
    "storno": ["annullo", "annulla", "storni e annulli", "cancellazione articolo", "storna"],
    "storno scontrino": [
        "annullo scontrino", "annulla scontrino", "cancellazione scontrino",
        "annullamento scontrino", "cancellare lo scontrino",
    ],
    # --- apertura / chiusura cassa ---
    "apertura cassa": [
        "apro la cassa", "apertura di cassa", "inizio giornata", "inizio turno",
        "accensione cassa", "fondo cassa", "apertura turno",
    ],
    "chiusura fiscale": [
        "chiusura giornaliera", "report z", "chiusura z", "azzeramento",
        "modalita z", "chiudo la cassa", "chiusura cassa", "fine giornata",
        "chiusura di cassa", "fine turno",
    ],
    "impulso cassetto": ["apertura cassetto", "apri cassetto", "cassetto rendiresto"],
    # --- report X / letture ---
    "letture": [
        "modalita x", "report x", "lettura x", "report intermedio",
        "lettura giornaliera", "lettura periodica", "letture statistiche",
    ],
    # --- resi ---
    "reso merce": ["reso parziale", "reso", "rimborso", "nota di credito", "pratica di reso"],
    # --- carta / rotolo ---
    "sostituzione rotolo": [
        "cambio rotolo", "cambio carta", "carta esaurita", "fine carta",
        "rotolo scontrino", "carta finita", "inserire rotolo", "rotolo",
        "rotolo carta",
    ],
    "giornale elettronico": [
        "dgfe", "mmc", "sostituzione giornale elettronico", "memoria dgfe",
        "scarico dgfe", "multi media card",
    ],
    # --- collegamento POS / periferiche ---
    "collegamento pos": [
        "abbinamento pos", "pairing pos", "configurazione pos", "collegare il pos",
        "terminale pagamento", "pagamento elettronico", "pos",
    ],
    "tipi pagamento": ["forme di pagamento", "pagamenti misti", "modalita di pagamento"],
    "connessione wifi": ["wifi", "rete wireless", "collegamento rete", "connessione di rete"],
    # --- fiscalizzazione / IVA ---
    "programmazione iva": [
        "aliquota", "aliquote iva", "impostazione iva", "reparti iva", "codice iva",
    ],
    "scorporo iva": ["scorporo", "imponibile", "stampa imponibile"],
    "obblighi fiscali": [
        "fiscalizzazione", "dichiarazione di installazione", "verifica periodica",
        "libretto fiscale", "conformita fiscale",
    ],
    # --- impostazioni base ---
    "data ora": [
        "data/ora", "data e ora", "impostazione data", "impostazione ora",
        "ora legale", "ora solare", "cambio ora", "orologio",
    ],
    "intestazione scontrino": [
        "logo scontrino", "ragione sociale scontrino", "testata scontrino",
        "piedino scontrino", "logo grafico", "logo",
    ],
    "blocco tastiera": ["tastiera bloccata", "sblocco tastiera"],
}

DOMAIN_VOCAB = sorted({term for k, v in SYNONYMS.items() for term in [k, *v]})

_VOCAB_BY_WORD_COUNT: dict[int, list[str]] = {}
for _term in DOMAIN_VOCAB:
    _VOCAB_BY_WORD_COUNT.setdefault(len(_term.split()), []).append(_term)

# Regex per estrazione di informazioni e token
MODEL_PATTERN = re.compile(r"\b(?:modello|mod\.?)\s*[:\-]?\s*([A-Za-z0-9\-]+)", re.IGNORECASE)
FIRMWARE_PATTERN = re.compile(
    r"\b(?:firmware|fw|versione)\s*[:\-]?\s*(v?[0-9]+(?:\.[0-9]+){0,2})", re.IGNORECASE
)
ERROR_CODE_PATTERN = re.compile(r"\b([A-Z])\s?-?\s?(\d{1,3})\b", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[0-9a-zà-ÿ]+", re.IGNORECASE)

# Parametri fuzzy
MIN_WINDOW_CHARS = 5
MIN_WINDOW_WORDS = 2
MAX_WINDOW_WORDS = 3
FUZZY_THRESHOLD = 88

# Parole funzionali della lingua
FUNCTION_WORDS = frozenset("""
    il lo la i gli le un uno una
    del dello della dei degli delle al allo alla ai agli alle
    dal dallo dalla dai dagli dalle nel nello nella nei negli nelle
    sul sullo sulla sui sugli sulle col coi
    di a da in con su per tra fra e o ed od
    che non si ci ne ha ho hai anno mi ti vi lo li
    ma se come cosa quando dove quale quali piu meno
""".split())

@dataclass
class NormalizedQuery:
    original: str
    corrected: str
    expanded: str
    model: str | None
    firmware: str | None
    matched_terms: list[str]
    error_codes: list[str] = field(default_factory=list)
    corrections: list[tuple[str, str]] = field(default_factory=list)


def _promuove_parola_funzionale(window: str, match: str) -> bool:
    """True se la correzione sostituirebbe una parola funzionale con un'altra
    parola: `lo scontrino` -> `logo scontrino`, `programmazione di` ->
    `programmazione iva`, `allo scontrino` -> `annullo scontrino`.

    Il confronto e' posizionale e non "la finestra contiene una parola
    funzionale": 14 termini del vocabolario ne contengono una (`nota di
    credito`, `data e ora`, `chiusura di cassa`, `cancellare lo scontrino`...)
    e devono restare correggibili. In quei casi la parola funzionale resta
    identica e cambia solo la parola piena, quindi la guardia non scatta.
    """
    wt, mt = window.split(), match.split()
    if len(wt) != len(mt):
        return False
    return any(a != b and a in FUNCTION_WORDS for a, b in zip(wt, mt))


def _best_vocab_match(window: str) -> str | None:
    """Miglior termine di dominio per questa finestra, o None se nessuno
    supera la soglia. Confronta solo contro termini con lo STESSO numero di
    parole della finestra (vedi nota FUZZY_*).

    Le finestre di una parola sola non vengono nemmeno confrontate: la
    guardia sta qui e non solo nel ciclo chiamante, cosi' vale per qualunque
    percorso arrivi a questa funzione.
    """
    word_count = len(window.split())
    if word_count < MIN_WINDOW_WORDS:
        return None
    candidates = _VOCAB_BY_WORD_COUNT.get(word_count)
    if not candidates or len(window) < MIN_WINDOW_CHARS:
        return None
    match = process.extractOne(
        window, candidates, scorer=fuzz.ratio, score_cutoff=FUZZY_THRESHOLD
    )
    if not match:
        return None
    return None if _promuove_parola_funzionale(window, match[0]) else match[0]


def _protected_spans(raw_query: str, tokens: list[str]) -> set[int]:
    """
    Codici di errore, modelli e firmware NON vanno toccati dalla correzione fuzzy"""
    protected: set[int] = set()
    raw_upper = raw_query.upper()
    codes = {f"{m.group(1)}{m.group(2)}".upper() for m in ERROR_CODE_PATTERN.finditer(raw_upper)}
    for i, tok in enumerate(tokens):
        if tok.upper() in codes or any(ch.isdigit() for ch in tok):
            protected.add(i)
    return protected


def _correct_tokens(tokens: list[str], protected: set[int]) -> tuple[list[str], list[tuple[str, str]]]:
    """Correzione dei refusi a livello di token, preservando l'ordine e la
    struttura della frase (serve a costruire `corrected`, che deve restare
    linguaggio naturale per il retrieval denso).

    Le finestre piu' lunghe vengono provate per prime: "strono scontrino"
    va corretto come unita' ("storno scontrino"), non come due parole
    indipendenti — il contesto e' esattamente cio' che rende la correzione
    affidabile.
    """
    result: list[str | None] = list(tokens)
    consumed: set[int] = set(protected)
    corrections: list[tuple[str, str]] = []

    for size in range(MAX_WINDOW_WORDS, MIN_WINDOW_WORDS - 1, -1):
        for i in range(len(tokens) - size + 1):
            span = range(i, i + size)
            if any(j in consumed for j in span):
                continue
            window = " ".join(tokens[i : i + size])
            match = _best_vocab_match(window)
            if not match:
                continue
            if match == window:
                consumed.update(span)
                continue
            result[i] = match
            for j in range(i + 1, i + size):
                result[j] = None
            consumed.update(span)
            corrections.append((window, match))

    return [t for t in result if t is not None], corrections


def _exact_domain_terms(query_lower: str) -> list[str]:
    """Termini di dominio presenti alla lettera nella query. Non passano dal
    fuzzy: se l'operatore ha scritto "collegamento pos", non c'e' nulla da
    indovinare."""
    found = []
    for term in DOMAIN_VOCAB:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, query_lower):
            found.append(term)
    return found


def _expand_with_synonyms(terms: list[str]) -> list[str]:
    """Per ogni termine riconosciuto, restituisce l'intera famiglia
    sinonimica a cui appartiene (canonico + tutte le varianti)."""
    expansion: list[str] = []
    for term in terms:
        for canonical, variants in SYNONYMS.items():
            if term == canonical or term in variants:
                expansion.extend([canonical, *variants])
                break
    return expansion


def normalize_query(raw_query: str) -> NormalizedQuery:
    # Estraggo informazioni di modello/firmware/errori
    model_match = MODEL_PATTERN.search(raw_query)
    fw_match = FIRMWARE_PATTERN.search(raw_query)
    error_codes = sorted(
        {f"{m.group(1).upper()}{m.group(2)}" for m in ERROR_CODE_PATTERN.finditer(raw_query)}
    )

    query_lower = raw_query.lower()
    tokens = _TOKEN_PATTERN.findall(query_lower) # Non uso split per non avere token sporchi
    protected = _protected_spans(raw_query, tokens)

    # Correggo la query
    corrected_tokens, corrections = _correct_tokens(tokens, protected)
    corrected = " ".join(corrected_tokens) if corrections else raw_query

    # Termini di dominio riconosciuti: quelli scritti correttamente
    matched = list(dict.fromkeys(_exact_domain_terms(query_lower) + [c[1] for c in corrections]))

    expansion_parts = [corrected, *_expand_with_synonyms(matched)]
    # I codici errore vanno ripetuti nella query BM25: sono discriminanti e
    # non devono essere annegati dai sinonimi aggiunti.
    expansion_parts.extend(error_codes)
    expanded = " | ".join(dict.fromkeys(p for p in expansion_parts if p))

    return NormalizedQuery(
        original=raw_query,
        corrected=corrected,
        expanded=expanded,
        model=model_match.group(1) if model_match else None,
        firmware=fw_match.group(1) if fw_match else None,
        matched_terms=matched,
        error_codes=error_codes,
        corrections=corrections,
    )
