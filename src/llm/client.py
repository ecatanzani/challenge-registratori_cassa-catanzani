"""
Factory per il chat model
"""

from functools import lru_cache

from config import settings
from src.audit.timing import timed


def get_chat_model():
    """Ritorna il chat model Anthropic configurato da .env.
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
    """Embedding multilingue locale.
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
