"""
Codice per il chunking delle pagine secondo la struttura a sezioni
"""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.pdf_parser import ImageRef, ParsedDocument
from src.ingestion.versioning import make_chunk_id

# Riconoscimento della struttura dei capitoli/sezioni/passi
SECTION_HEADER_RE = re.compile(
    r"^(?:[1-9]\d?(?:\.\d{1,2}){0,2}[.)]?\s+[A-ZÀ-Ù].{3,80}|"
    r"(?:CAPITOLO|Capitolo|SEZIONE|Sezione)\s+\d+.{0,80}|"
    r"(?:Passo|PASSO)\s+\d+.{0,80})$",
    re.MULTILINE,
)

# Riconoscimento del sommario
TOC_LEADER_RE = re.compile(r"\.{4,}")

CHUNK_SIZE = 900            # tetto di caratteri per chunk
CHUNK_OVERLAP = 150         # sovrapposizione quando un pezzo va spezzato
TOC_LINE_RATIO = 0.35       # oltre 1/3 di righe puntinate = sommario
MIN_CHUNK_CHARS = 30        # sotto, il chunk non può rispondere a niente

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    section_title: str | None
    text: str
    image_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _looks_like_table_of_contents(text: str) -> bool:
    """Controllo se il testo e' (in prevalenza) un indice dei contenuti.
    """
    righe = [ln for ln in text.split("\n") if ln.strip()]
    if len(righe) < 4:
        return False
    puntinate = sum(1 for ln in righe if TOC_LEADER_RE.search(ln))
    return puntinate / len(righe) >= TOC_LINE_RATIO


def _split_page_by_headers(text: str) -> list[tuple[str | None, str]]:
    """Divide il testo di una pagina in (titolo_sezione, corpo) usando gli
    header trovati.

    Ogni segmento è una tupla di "Nome Sezione", "testo". Il preambolo non ha nome sezione.
    """
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [(None, text)]

    segments: list[tuple[str | None, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        segments.append((None, preamble))

    for i, m in enumerate(matches):
        title = m.group().strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            segments.append((title, body))
    if not segments:
        return [(None, text)]
    return segments


def _norm_titolo(titolo: str | None) -> str:
    """Chiave di confronto fra titoli. Collassa spazi e a capo."""
    return " ".join((titolo or "").split())

def _titoli_con_y(page_content) -> list[tuple[float, str]]:
    """(y, titolo) per i titoli di sezione presenti sulla pagina.
    """
    righe = page_content.text_lines
    if not righe:
        return []
    trovati: list[tuple[float, str]] = []
    for titolo in SECTION_HEADER_RE.findall(page_content.text):
        prima_riga = titolo.strip().split("\n")[0].strip()
        y = next((y for y, testo in righe if testo == prima_riga), None)
        if y is not None:
            trovati.append((y, titolo.strip()))
    return sorted(trovati, key=lambda t: t[0])


def _figure_per_sezione(parsed: ParsedDocument) -> dict[str, list[str]]:
    """titolo di sezione -> image_id delle figure che le appartengono.
    """
    per_sezione: dict[str, list[str]] = {}
    ultimo_titolo: str | None = None

    for page_content in parsed.pages:
        titoli = _titoli_con_y(page_content)
        for immagine in page_content.images:
            y_figura = immagine.bbox[1]
            sopra = [t for y, t in titoli if y <= y_figura]
            sezione = sopra[-1] if sopra else ultimo_titolo
            chiave = _norm_titolo(sezione)
            # Una figura senza sezione (copertina) non va assegnata
            if not chiave:
                continue
            per_sezione.setdefault(chiave, []).append(immagine.image_id)
        if titoli:
            ultimo_titolo = titoli[-1][1]

    return per_sezione

def chunk_document(
    parsed: ParsedDocument,
    version: str,
    model: str | None = None,
    firmware: str | None = None,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " "]
    )

    chunks: list[Chunk] = []
    counter = 0
    # Ultimo titolo di sezione incontrato, che sopravvive al cambio pagina.
    # Permette di citare come "p.15 - Inizializzazione DGFE" senza perdere il titolo
    last_title: str | None = None

    figure_per_sezione = _figure_per_sezione(parsed)

    for page_content in parsed.pages:
        segments = _split_page_by_headers(page_content.text)
        for seg_index, (title, body) in enumerate(segments):
            if _looks_like_table_of_contents(body):
                continue

            # Segmento senza header proprio: se e' il primo della pagina, e'
            # la coda della sezione precedente e ne eredita il titolo.
            is_continuation = title is None and seg_index == 0 and last_title is not None
            if is_continuation:
                title = last_title
            elif title is not None:
                last_title = title

            image_ids_sezione = figure_per_sezione.get(_norm_titolo(title), [])

            if len(body) <= CHUNK_SIZE:
                sub_texts = [body]
            else:
                sub_texts = splitter.split_text(body)

            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue
                if len(sub_text.strip()) < MIN_CHUNK_CHARS:
                    continue
                if _looks_like_table_of_contents(sub_text):
                    continue
                counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(parsed.doc_id, version, counter),
                        doc_id=parsed.doc_id,
                        page_start=page_content.page,
                        page_end=page_content.page,
                        section_title=title,
                        text=sub_text.strip(),
                        image_ids=image_ids_sezione,
                        metadata={
                            "version": version,
                            "model": model,
                            "firmware": firmware,
                            "ocr_used": page_content.ocr_used,
                            "source_path": parsed.source_path,
                            "is_continuation": is_continuation,
                        },
                    )
                )
    return chunks
