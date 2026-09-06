"""
Costruisce e persiste l'indice di retrieval:
- Chroma (dense vector store) per similarita' semantica.
- Un file JSONL con tutti i chunk (usato per ricostruire il BM25Retriever
  a runtime: BM25 non ha bisogno di essere "addestrato" pesantemente ma
  serve tenere il corpus disponibile).
- SQLite per i metadati che non appartengono ne' al vettore ne' al testo:
    * `manuals` — versioning documenti (doc_id, versione, modello, firmware,
      data ingestion), usato dal tie-break "preferisci la versione piu'
      recente" e dall'audit trail;
    * `images`  — bounding box e area di ogni figura estratta dal PDF.
"""

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from src.ingestion.chunker import Chunk
from src.ingestion.pdf_parser import ImageRef
from src.llm.client import get_embedding_model

# Metrica di distanza esplicita per la collection Chroma: coseno, cosi' il
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# Telemetria Chroma disattivata: l'ingestion gira su manuali di clienti, non
# spedisce eventi d'uso a servizi terzi. Effetto collaterale utile: silenzia i
# "Failed to send telemetry event ..." dovuti all'incompatibilita' di firma tra
# chromadb e le versioni recenti di posthog. Deve essere identica su tutti i
# client dello stesso processo, altrimenti Chroma rifiuta la seconda istanza.
CHROMA_CLIENT_SETTINGS = ChromaSettings(anonymized_telemetry=False, is_persistent=True)


def db_path() -> Path:
    return settings.vectorstore_dir / "manuals.db"


def _chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    docs = []
    for c in chunks:
        docs.append(
            Document(
                page_content=c.text,
                metadata={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_title": c.section_title or "",
                    "image_ids": json.dumps(c.image_ids),
                    **{k: v for k, v in c.metadata.items() if v is not None},
                },
            )
        )
    return docs


def _init_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS manuals (
            doc_id TEXT NOT NULL,
            version TEXT NOT NULL,
            source_path TEXT,
            model TEXT,
            firmware TEXT,
            n_chunks INTEGER,
            n_pages INTEGER,
            n_ocr_pages INTEGER,
            n_ocr_failed_pages INTEGER,
            ingested_at TEXT,
            PRIMARY KEY (doc_id, version)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            version TEXT NOT NULL,
            page INTEGER NOT NULL,
            x0 REAL, y0 REAL, x1 REAL, y1 REAL,
            area REAL,
            PRIMARY KEY (image_id, version)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_page ON images (doc_id, version, page)")
    conn.commit()
    return conn


def index_chunks(
    chunks: list[Chunk],
    doc_id: str,
    version: str,
    model: str | None,
    firmware: str | None,
    images: list[ImageRef] | None = None,
    n_pages: int = 0,
    n_ocr_pages: int = 0,
    n_ocr_failed_pages: int = 0,
) -> None:
    from datetime import datetime, timezone

    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    documents = _chunks_to_documents(chunks)

    # --- Dense (Chroma) ---
    embeddings = get_embedding_model()
    vectorstore = Chroma(
        collection_name="manuali_cassa",
        embedding_function=embeddings,
        persist_directory=str(settings.vectorstore_dir),
        collection_metadata=COLLECTION_METADATA,
        client_settings=CHROMA_CLIENT_SETTINGS,
    )
    # Re-ingestion idempotente
    existing = vectorstore.get(
        where={"$and": [{"doc_id": doc_id}, {"version": version}]}
    )
    if existing and existing.get("ids"):
        vectorstore.delete(ids=existing["ids"])
    vectorstore.add_documents(documents, ids=[c.chunk_id for c in chunks])

    # --- Corpus JSONL per BM25 (ricostruito a runtime, vedi hybrid_retriever.py) ---
    corpus_path = settings.vectorstore_dir / "bm25_corpus.jsonl"
    existing_lines = []
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                meta = row["metadata"]
                if not (meta.get("doc_id") == doc_id and meta.get("version") == version):
                    existing_lines.append(row)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in existing_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for d in documents:
            f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")

    # --- Metadati versioning / audit / bounding box (SQLite) ---
    conn = _init_sqlite(db_path())
    conn.execute(
        """
        INSERT INTO manuals (doc_id, version, source_path, model, firmware, n_chunks,
                             n_pages, n_ocr_pages, n_ocr_failed_pages, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id, version) DO UPDATE SET
            source_path=excluded.source_path, model=excluded.model,
            firmware=excluded.firmware, n_chunks=excluded.n_chunks,
            n_pages=excluded.n_pages, n_ocr_pages=excluded.n_ocr_pages,
            n_ocr_failed_pages=excluded.n_ocr_failed_pages,
            ingested_at=excluded.ingested_at
        """,
        (
            doc_id, version, chunks[0].metadata.get("source_path", "") if chunks else "",
            model, firmware, len(chunks), n_pages, n_ocr_pages, n_ocr_failed_pages,
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.execute("DELETE FROM images WHERE doc_id = ? AND version = ?", (doc_id, version))
    for img in images or []:
        x0, y0, x1, y1 = img.bbox
        conn.execute(
            "INSERT OR REPLACE INTO images (image_id, doc_id, version, page, x0, y0, x1, y1, area)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (img.image_id, doc_id, version, img.page, x0, y0, x1, y1, img.area),
        )
    conn.commit()
    conn.close()


@lru_cache(maxsize=1)
def load_vectorstore() -> Chroma:
    """Cache a livello di processo: riaprire la connessione a Chroma da disco
    ha un costo minore rispetto a ricaricare l'embedding model, ma non e'
    zero — la cache evita comunque di ripeterlo ad ogni domanda. Nota: se un
    altro processo scrive nell'indice (es. una nuova ingestion) mentre l'app
    e' in esecuzione, questa cache non lo vede finche' il processo non viene
    riavviato — accettabile qui perche' ingestion e serving sono comandi
    separati (vedi README, sezione Setup).
    """
    embeddings = get_embedding_model()
    return Chroma(
        collection_name="manuali_cassa",
        embedding_function=embeddings,
        persist_directory=str(settings.vectorstore_dir),
        collection_metadata=COLLECTION_METADATA,
        client_settings=CHROMA_CLIENT_SETTINGS,
    )


def index_is_ready() -> bool:
    """True se esiste un indice interrogabile. Serve alla UI per distinguere
    'nessuna risposta trovata' da 'non hai ancora lanciato l'ingestion', che
    per chi valuta il progetto sono due situazioni molto diverse."""
    try:
        return load_vectorstore()._collection.count() > 0
    except Exception:
        return False
