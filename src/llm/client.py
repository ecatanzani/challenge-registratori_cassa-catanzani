"""
Factory per il chat model
"""
from __future__ import annotations

from functools import lru_cache

from config import settings
from src.audit.timing import timed


def get_chat_model():
    """Ritorna il chat model Anthropic configurato da .env.

    Nessun parametro di campionamento: sui modelli Claude della generazione
    corrente (famiglia 5, e 4.7/4.8) `temperature`, `top_p` e `top_k` sono
    stati rimossi dall'API e una richiesta che li contiene viene rifiutata con
    400 `temperature is deprecated for this model`. La determinismo che qui
    serviva non e' comunque affidato al campionamento ma all'output
    strutturato: la risposta e' un oggetto Pydantic validato (StructuredAnswer)
    e i chunk_id citati vengono verificati contro quelli realmente recuperati
    in _validate_grounding().
    """
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY non impostata. Compila .env (vedi .env.example)."
        )
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        max_tokens=2048,
    )


class _E5PrefixedEmbeddings:
    """Wrapper che antepone i prefissi richiesti dai modelli della famiglia E5.

    `intfloat/multilingual-e5-*` NON e' un encoder simmetrico: e' addestrato
    con due prefissi obbligatori, `"query: "` davanti alla domanda e
    `"passage: "` davanti al documento. Usarlo senza — come faceva la versione
    precedente — non produce un errore: produce un degrado silenzioso in cui
    tutte le similarita' si comprimono in una banda alta e stretta.

    Misurato su questo indice, prima della correzione: le domande fuori tema
    ottenevano 0.777-0.812 di similarita' col chunk piu' vicino, quelle in
    tema 0.837-0.887. Le due classi restano separabili, ma con un margine di
    0.024 — cioe' una soglia del gate di dominio talmente sottile da non
    reggere nessuna domanda nuova. Il problema non era la soglia: era che al
    modello si stava chiedendo di lavorare fuori dal suo contratto d'uso.

    Il wrapper si applica SOLO ai modelli E5 (`_needs_e5_prefixes`): mettere
    "query:" davanti all'input di un modello che non lo prevede sarebbe
    l'errore speculare, e cambiare EMBEDDING_MODEL in .env non deve
    trascinarsi dietro un prefisso sbagliato.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(self, inner):
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([self.PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(self.QUERY_PREFIX + text)

    def __getattr__(self, name):  # deleghe (LangChain ne usa altre in casi specifici)
        return getattr(self._inner, name)


def _needs_e5_prefixes(model_name: str) -> bool:
    return "e5" in model_name.lower()


@lru_cache(maxsize=1)
def get_embedding_model():
    """Embedding multilingue locale (nessuna chiave richiesta, gira offline).
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    with timed("embedding.caricamento_modello"):
        model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    if _needs_e5_prefixes(settings.embedding_model):
        return _E5PrefixedEmbeddings(model)
    return model
