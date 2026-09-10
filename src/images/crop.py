import hashlib
import sqlite3
from pathlib import Path

import pymupdf

from config import settings
from src.generation.schemas import ImageCitation
from src.ingestion.indexer import db_path
from src.ingestion.versioning import parse_version

CROP_RENDER_DPI = 200
PAGE_PREVIEW_DPI = 110
HIGHLIGHT_MARGIN = 4
HIGHLIGHT_COLOR = (0.85, 0.10, 0.10)
HIGHLIGHT_WIDTH = 2.0

MIN_MEANINGFUL_IMAGE_AREA = 3000.0


def _query(sql: str, params: tuple) -> list[tuple]:
    path = db_path()
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _latest_source_path(doc_id: str) -> str | None:
    """Percorso del PDF per la versione piu' recente di questo doc_id."""
    rows = _query("SELECT version, source_path FROM manuals WHERE doc_id = ?", (doc_id,))
    if not rows:
        return None
    return max(rows, key=lambda r: parse_version(r[0]))[1]


def _image_records(
    doc_id: str, image_ids: list[str]
) -> dict[str, tuple[tuple[float, float, float, float], float, int]]:
    """image_id -> (bbox, area, page), dalla versione piu' recente di ciascuna.
    """
    if not image_ids:
        return {}
    placeholders = ",".join("?" for _ in image_ids)
    rows = _query(
        f"SELECT image_id, version, x0, y0, x1, y1, area, page FROM images"
        f" WHERE doc_id = ? AND image_id IN ({placeholders})",
        (doc_id, *image_ids),
    )
    best: dict[str, tuple] = {}
    seen_version: dict[str, tuple[int, ...]] = {}
    for image_id, version, x0, y0, x1, y1, area, page in rows:
        v = parse_version(version)
        if image_id not in seen_version or v > seen_version[image_id]:
            seen_version[image_id] = v
            best[image_id] = ((x0, y0, x1, y1), area or 0.0, page)
    return best


def select_relevant_image_ids(doc_id: str, candidate_image_ids: list[str]) -> list[str]:
    """TUTTE le figure pertinenti fra quelle della SEZIONE del chunk,
    ordinate dalla piu' grande alla piu' piccola.
    """
    if not candidate_image_ids:
        return []

    records = _image_records(doc_id, candidate_image_ids)
    if not records:
        # Indice costruito da una versione precedente del codice, senza la
        # tabella images: si degrada al primo candidato invece di sollevare.
        return candidate_image_ids[:1]

    ranked = sorted(records.items(), key=lambda kv: kv[1][1], reverse=True)
    meaningful = [iid for iid, (_, area, _) in ranked if area >= MIN_MEANINGFUL_IMAGE_AREA]
    return meaningful or [ranked[0][0]]


def _preview_path(doc_id: str, page: int, image_ids: list[str]) -> Path:
    """Percorso dell'anteprima di pagina"""
    digest = hashlib.sha1("|".join(sorted(image_ids)).encode("utf-8")).hexdigest()[:8]
    return settings.crops_dir / f"{doc_id}_p{page}_hl{digest}.png"


def _render_crop(page: pymupdf.Page, bbox: pymupdf.Rect, out_path: Path) -> None:
    clip = pymupdf.Rect(
        max(bbox.x0 - HIGHLIGHT_MARGIN, 0),
        max(bbox.y0 - HIGHLIGHT_MARGIN, 0),
        min(bbox.x1 + HIGHLIGHT_MARGIN, page.rect.x1),
        min(bbox.y1 + HIGHLIGHT_MARGIN, page.rect.y1),
    )
    zoom = CROP_RENDER_DPI / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip)
    pix.save(str(out_path))


def _render_page_preview(page: pymupdf.Page, bboxes: list[pymupdf.Rect], out_path: Path) -> None:
    """Selezione della zona interessata con una bbox rossa
    """
    for bbox in bboxes:
        page.draw_rect(bbox, color=HIGHLIGHT_COLOR, width=HIGHLIGHT_WIDTH)
    zoom = PAGE_PREVIEW_DPI / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    pix.save(str(out_path))


def crop_images_for_chunk(
    doc_id: str, page: int, candidate_image_ids: list[str]
) -> list[ImageCitation]:
    """Ritagli di tutte le figure pertinenti del chunk + un'anteprima per
    ciascuna pagina toccata, che evidenzia le figure presenti su quella.
    """
    relevant_ids = select_relevant_image_ids(doc_id, candidate_image_ids)
    if not relevant_ids:
        return []

    records = _image_records(doc_id, relevant_ids)
    bboxes = {iid: records[iid][0] for iid in relevant_ids if iid in records}
    if not bboxes:
        return []

    source_path = _latest_source_path(doc_id)
    if not source_path or not Path(source_path).exists():
        return []

    # Raggruppate per la LORO pagina: una apertura e un rendering per pagina.
    ids_per_pagina: dict[int, list[str]] = {}
    for iid in bboxes:
        #(bbox, area, page) per ogni riga di record (page è un fallback se il database non ritorna nulla)
        ids_per_pagina.setdefault(records[iid][2] or page, []).append(iid)

    settings.crops_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = {iid: settings.crops_dir / f"{iid}.png" for iid in bboxes}
    preview_paths = {
        num: _preview_path(doc_id, num, ids) for num, ids in ids_per_pagina.items()
    }

    da_renderizzare = {
        num: ids
        for num, ids in ids_per_pagina.items()
        if not preview_paths[num].exists() or any(not crop_paths[i].exists() for i in ids)
    }
    if da_renderizzare:
        try:
            doc = pymupdf.open(source_path)
            try:
                for num, ids in da_renderizzare.items():
                    pdf_page = doc[num - 1]
                    for iid in ids:
                        if not crop_paths[iid].exists():
                            _render_crop(pdf_page, pymupdf.Rect(*bboxes[iid]), crop_paths[iid])
                    if not preview_paths[num].exists():
                        # disegna i riquadri sulla pagina in memoria
                        _render_page_preview(
                            pdf_page,
                            [pymupdf.Rect(*bboxes[iid]) for iid in ids],
                            preview_paths[num],
                        )
            finally:
                doc.close()
        except Exception:
            return []

    return [
        ImageCitation(
            image_id=iid,
            doc_id=doc_id,
            page=num,
            bbox=bboxes[iid],
            crop_path=str(crop_paths[iid]),
            page_preview_path=(
                str(preview_paths[num]) if preview_paths[num].exists() else None
            ),
            highlighted_on_page=len(ids),
        )
        for num, ids in ids_per_pagina.items()
        for iid in ids
        if crop_paths[iid].exists()
    ]
