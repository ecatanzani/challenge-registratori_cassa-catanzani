"""
Configurazione centrale dell'applicazione.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-large"

    # Percorsi
    manuals_dir: Path = Path("./data/manuals")
    vectorstore_dir: Path = Path("./data/vectorstore")
    crops_dir: Path = Path("./data/crops")
    audit_log_path: Path = Path("./data/audit_log.jsonl")

    # Retrieval
    top_k_dense: int = 8            # candidati richiesti dalla ricerca semantica
    top_k_bm25: int = 8             # candidati richiesti dalla ricerca lessicale
    top_k_final: int = 5            # candidati richiesti che finiscono nel context del modello
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    rerank_enabled: bool = True

    # Threshold per il filtro di più modelli di cassa
    ambiguity_relative_margin: float = 0.85

    
    max_images_per_answer: int = 12
    off_topic_similarity_threshold: float = 0.8 # threshold calibrata sulle domande in tema
    debug_timing_logs: bool = True


settings = Settings()

# Inizializzazione cartelle
for directory in (settings.manuals_dir, settings.vectorstore_dir, settings.crops_dir):
    directory.mkdir(parents=True, exist_ok=True)
settings.audit_log_path.parent.mkdir(parents=True, exist_ok=True)