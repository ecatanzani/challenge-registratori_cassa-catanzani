"""
Parsing dei manuali PDF: estrae testo e immagini (con bounding box) pagina
per pagina. Se una pagina ha pochissimo testo estraibile (tipico di uno
scan), applica OCR di fallback con pytesseract sul rendering della pagina.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf

# OCR params
MIN_CHARS_FOR_NATIVE_TEXT = 40  # Numero minimo di caratteri sotto i quali una pagine è considerata scannerizzata
MIN_CHARS_FOR_USEFUL_OCR = 20   # Numero minimo di caratteri per un OCR
OCR_RENDER_DPI = 300
OCR_LANGUAGES = "ita+eng"

# Bolierplate params
BOILERPLATE_MIN_PAGE_RATIO = 0.5  # riga presente su almeno meta' delle pagine
BOILERPLATE_MAX_LINE_CHARS = 120  # un header/footer e' corto; un paragrafo no

# Immagini
MIN_IMAGE_AREA = 100

@dataclass
class ImageRef:
    image_id: str
    doc_id: str
    page: int
    bbox: tuple[float, float, float, float]
    ocr_used: bool = False

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class PageContent:
    page: int
    text: str
    ocr_used: bool
    images: list[ImageRef] = field(default_factory=list)
    text_lines: list[tuple[float, str]] = field(default_factory=list)
    ocr_attempted: bool = False
    ocr_failed: bool = False
    ocr_empty: bool = False
    ocr_error: str | None = None


@dataclass
class ParsedDocument:
    doc_id: str
    source_path: str
    pages: list[PageContent]

    @property
    def images(self) -> list[ImageRef]:
        return [img for p in self.pages for img in p.images]

OcrStatus = Literal["ok", "empty", "unavailable"]

def _ocr_page(page: pymupdf.Page) -> tuple[str, OcrStatus, str | None]:
    """OCR di fallback per pagine scansionate con poco testo nativo.

    Ritorna `(testo, stato, dettaglio)`. Gli stati sono TRE e non due, perche'
    due situazioni che prima finivano nello stesso flag non hanno niente in
    comune:

        "ok"           l'OCR ha girato e ha prodotto testo utilizzabile.

        "empty"        l'OCR ha girato benissimo, ma sulla pagina c'era
                       pochissimo da leggere. E' il caso di copertine,
                       frontespizi e separatori: NON e' un guasto e non
                       richiede nessun intervento.

        "unavailable"  l'OCR non ha potuto girare: binario tesseract assente,
                       pacchetto lingua mancante, pagina corrotta. Questo si'
                       che e' un problema, e va segnalato come tale.
    """
    try:
        import io

        import pytesseract
        from PIL import Image

        pix: pymupdf.Pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI) # Trasformazione della pagina PDF in un pixels per tesseract
        img: Image = Image.open(io.BytesIO(pix.tobytes("png")))
        text: str = pytesseract.image_to_string(img, lang=OCR_LANGUAGES).strip()
    except Exception as exc:
        return "", "unavailable", f"{type(exc).__name__}: {exc}"

    if len(text) < MIN_CHARS_FOR_USEFUL_OCR:
        return "", "empty", f"pagina quasi priva di testo ({len(text)} caratteri letti)"
    return text, "ok", None


def _detect_boilerplate_lines(page_texts: list[str]) -> set[str]:
    """Righe di intestazione/pie' di pagina ricorrenti, rilevate per FREQUENZA.

    Se una riga presente su piu' della meta' delle
    pagine viene considerata come impaginazione, non contenuto.
    """
    from collections import Counter

    if not page_texts:
        return set()

    frequency: Counter[str] = Counter()
    for text in page_texts:
        for line in {ln.strip() for ln in text.split("\n") if ln.strip()}:
            if len(line) <= BOILERPLATE_MAX_LINE_CHARS:
                frequency[line] += 1

    soglia = len(page_texts) * BOILERPLATE_MIN_PAGE_RATIO
    return {line for line, n in frequency.items() if n > soglia} # Set permette di evitare che una tabella con righe ripetute sia considerata come testo boilerplate


def _strip_boilerplate(text: str, boilerplate: set[str], page_num: int) -> str:
    """Toglie le righe ricorrenti e il numero di pagina isolato (non rimuovibile per frequenza).
    """
    attesi = {str(page_num), str(page_num - 1), str(page_num + 1)}
    tenute = [
        ln for ln in text.split("\n")
        if ln.strip() not in boilerplate and ln.strip() not in attesi
    ]
    return "\n".join(tenute).strip()

def _lines_with_y(page: pymupdf.Page, boilerplate: set[str], page_num: int) -> list[tuple[float, str]]:
    """Righe della pagina con la loro coordinata verticale, ordinate dall'alto.
    """
    attesi = {str(page_num), str(page_num - 1), str(page_num + 1)}
    righe: list[tuple[float, str]] = []
    for blocco in page.get_text("dict")["blocks"]:
        for riga in blocco.get("lines", []):
            testo = "".join(span["text"] for span in riga["spans"]).strip()
            if testo and testo not in boilerplate and testo not in attesi:
                righe.append((riga["bbox"][1], testo))
    return sorted(righe, key=lambda r: r[0])

def parse_pdf(path: str | Path, doc_id: str, page_shift: int = 1) -> ParsedDocument:
    path = Path(path)
    doc = pymupdf.open(path)
    pages: list[PageContent] = []

    # Riconoscimento del testo ricorrente
    raw_texts = [doc[i].get_text("text") for i in range(len(doc))]
    boilerplate = _detect_boilerplate_lines(raw_texts)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + page_shift  # Risolve il problema dello shift di pagine nel manuale

        native_text = _strip_boilerplate(raw_texts[page_index], boilerplate, page_num)
        text_lines = _lines_with_y(page, boilerplate, page_num)

        # Inizializzo le variabili che descriveranno la classe PageContent.
        ocr_attempted = False
        ocr_used = False
        ocr_failed = False
        ocr_empty = False
        ocr_error = None
        text = native_text

        # Controllo se serve un fallback con tecnica OCR
        if len(native_text) < MIN_CHARS_FOR_NATIVE_TEXT:
            ocr_attempted = True
            ocr_text, ocr_status, ocr_error = _ocr_page(page)
            if ocr_status == "ok":
                ocr_used = True
                if native_text and native_text not in ocr_text:
                    text = (native_text + "\n" + ocr_text).strip()
                else:
                    text = ocr_text
            elif ocr_status == "empty":
                ocr_empty = True
            else:
                ocr_failed = True

        # Cerco immagini nella pagina
        images: list[ImageRef] = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            rects = page.get_image_rects(xref)
            for rect_index, rect in enumerate(rects):
                image_id = f"{doc_id}_p{page_num}_img{img_index}_{rect_index}"
                immagine = ImageRef(
                    image_id=image_id,
                    doc_id=doc_id,
                    page=page_num,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    ocr_used=ocr_used, # Sapere se la pagina che contiene l'immagine era una scansione
                )
                # Gli indici restano quelli di enumerate: un image_id non cambia
                # significato per il fatto che una scheggia accanto sia stata scartata.
                if immagine.area < MIN_IMAGE_AREA:
                    continue
                images.append(immagine)

        pages.append(
            PageContent(
                page=page_num,
                text=text,
                ocr_used=ocr_used,
                images=images,
                text_lines=text_lines,
                ocr_attempted=ocr_attempted,
                ocr_empty=ocr_empty,
                ocr_failed=ocr_failed,
                ocr_error=ocr_error,
            )
        )

    doc.close()
    return ParsedDocument(doc_id=doc_id, source_path=str(path), pages=pages)
