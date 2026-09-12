"""
Normalizzazione della query utente prima del retrieval. Produce:

- `original`  : domanda originale dell'utente.
- `corrected` : domanda con i refusi corretti sulle parole del manuale
                indicizzato ("rololo di crata" -> "rotolo di carta"). La forma della
                domanda è la medesima e verrà utilizzata dal retriever denso
- `expanded`  : domanda corretta piu' tutte le varianti sinonimiche note
                ("annullo scontrino | storno scontrino | ...") per il retriever BM25
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz import distance, fuzz, process

from config import settings

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

# Correzione dei refusi, parola per parola
MIN_CHARS_CORREZIONE = 5   # sotto, una lettera cambia troppo il significato
SOGLIA_ZIPF = 1.0          # quanto una parola deve essere comune per valere come italiana
MAX_DISTANZA_CORTE = 1     # parole fino a 6 lettere: un solo errore
MAX_DISTANZA_LUNGHE = 2    # oltre, se ne tollerano due

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


@lru_cache(maxsize=1)
def _parole_del_manuale() -> Counter:
    """Parole dei chunk indicizzati, con la loro frequenza: e' il bersaglio
    della correzione.
    """
    freq: Counter = Counter()
    corpus = settings.vectorstore_dir / "bm25_corpus.jsonl"
    if not corpus.exists():
        return freq
    with open(corpus, "r", encoding="utf-8") as f:
        for riga in f:
            freq.update(_TOKEN_PATTERN.findall(json.loads(riga)["text"].lower()))
    return freq


@lru_cache(maxsize=1)
def _italiano() -> dict[str, float]:
    """Parole italiane con la loro frequenza (scala zipf), da `wordfreq`.

    Serve a due cose: sapere che una parola esiste in italiano ed
    offrire un candidato quando la parola giusta nel manuale non c'e'.
    """
    from wordfreq import top_n_list, zipf_frequency

    return {parola: zipf for parola in top_n_list("it", 300000) if parola.isalpha()
            if (zipf := zipf_frequency(parola, "it")) >= SOGLIA_ZIPF}


def _correggi_parola(token: str) -> str:
    """Parola con cui sostituire un refuso, o il token invariato.

    Si corregge solo cio' che non e' ne' una parola del manuale ne' una parola
    italiana.

    I candidati vengono da due parti, con priorita' diverse:

    - il manuale vince a parita' di distanza, perche' e' la parola che BM25 puo'
      davvero agganciare: `strono` diventa `storno` e non `stronco`, italiano e
      alla stessa distanza;
    - l'italiano vince se offre una parola STRETTAMENTE piu' vicina, perche'
      quando la parola giusta nel manuale non esiste conservare il significato
      vale piu' di un aggancio lessicale;
    - se il manuale non ha candidati si prende comunque l'italiano. Non aiuta il retrieval, ma l'operatore vede la sua
      domanda scritta correttamente.
    """
    freq = _parole_del_manuale()
    italiano = _italiano()
    if (not freq or token in freq or len(token) < MIN_CHARS_CORREZIONE
            or any(ch.isdigit() for ch in token) or token in italiano):
        return token

    limite = MAX_DISTANZA_CORTE if len(token) <= 6 else MAX_DISTANZA_LUNGHE
    dal_manuale, distanza_manuale = _piu_vicina(token, list(freq), limite, freq.__getitem__)
    dall_italiano, distanza_italiano = _piu_vicina(
        token, [w for w in italiano if abs(len(w) - len(token)) <= limite], limite, italiano.__getitem__)

    if dal_manuale is None:
        return dall_italiano or token
    if dall_italiano is not None and distanza_italiano < distanza_manuale:
        return dall_italiano
    return dal_manuale


def _piu_vicina(token: str, candidati: list[str], limite: int, frequenza) -> tuple[str | None, int | None]:
    """Candidato piu' vicino entro `limite`; a parita' di distanza vince quello
    piu' frequente secondo `frequenza`."""
    vicine = process.extract(token, candidati, scorer=distance.DamerauLevenshtein.distance,
                             score_cutoff=limite, limit=None)
    if not vicine:
        return None, None
    minima = min(d for _, d, _ in vicine)
    # Ritornare la parola più frequente a parità di distanza
    return max((parola for parola, d, _ in vicine if d == minima), key=frequenza), minima


def _correct_tokens(tokens: list[str], protected: set[int]) -> tuple[list[str], list[tuple[str, str]]]:
    """Correzione dei refusi token per token, preservando l'ordine e la
    struttura della frase (serve a costruire `corrected`, che deve restare
    linguaggio naturale per il retrieval denso)."""
    corretti = [t if i in protected else _correggi_parola(t) for i, t in enumerate(tokens)]
    corrections = [(prima, dopo) for prima, dopo in zip(tokens, corretti) if prima != dopo]
    return corretti, corrections


def _termini_riconosciuti(tokens: list[str]) -> list[str]:
    """Termini di dominio riconosciuti a meno di refusi, per l'espansione
    sinonimica di BM25.

    Le finestre piu' lunghe vengono provate per prime, cosi' "carta esaurito" viene riconosciuto come
    `carta esaurita` invece che pezzo per pezzo.
    """
    trovati: list[str] = []
    consumati: set[int] = set()
    for size in range(MAX_WINDOW_WORDS, MIN_WINDOW_WORDS - 1, -1):
        for i in range(len(tokens) - size + 1):
            span = range(i, i + size)
            if any(j in consumati for j in span):
                continue
            match = _best_vocab_match(" ".join(tokens[i : i + size]))
            if match:
                trovati.append(match)
                consumati.update(span)
    return trovati


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

    # Termini di dominio riconosciuti: quelli scritti alla lettera piu' quelli
    # riconosciuti a meno di refusi. Serve all'espansione sinonimica.
    testo_corretto = " ".join(corrected_tokens)
    matched = list(dict.fromkeys(
        _exact_domain_terms(testo_corretto) + _termini_riconosciuti(corrected_tokens)))

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
