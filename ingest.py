"""
Codice per l'ingestion dei dati e la creazione del vector-store ed i database iniziali.
Il codice parsa il documento PDF ed estrae tutte le informazioni rilevanti per la successiva ricerca.
"""

import argparse

from src.logging.logger import setup_main_logger
from src.ingestion.chunker import chunk_document
from src.ingestion.indexer import index_chunks
from src.ingestion.pdf_parser import parse_pdf


MAIN_LOGGER_NAME: str = "ingest"

def parse_args() -> argparse.Namespace:
    """
    Funzione per il parsing degli argomenti da linea di comando.

    Riferimento al documento
   --doc-id     -->   identificativo del manuale. Viene richiamato dei chunk
   --version    -->   versione del manuale, è come se fosse l'asse temporale e la coesistenza di manuali di versioni differenti

   Riferimento al dispositivo documentato
   --model      -->   a quale hardware si applica il contenuto
   --firmware   -->   a quale revisione software di quell'hardware
    """
    parser = argparse.ArgumentParser(description="Ingestion di un manuale PDF nell'indice RAG.")
    parser.add_argument("--pdf", required=True, help="Percorso del file PDF")
    parser.add_argument("--doc-id", required=True, help="ID univoco del documento (es. stampante_x1)")
    parser.add_argument("--version", required=True, help="Versione del manuale (es. 1.2)")
    parser.add_argument(
        "--model", default=None,
        help="Modello del dispositivo documentato (es. X1). Filtra il retrieval quando "
                "l'operatore nomina un modello nella domanda, e alimenta la disambiguazione "
                "quando piu' modelli risultano ugualmente plausibili.")
    parser.add_argument(
        "--firmware", default=None,
        help="Versione firmware documentata (es. 2.3.1). Oggi e' un metadato di "
                "tracciabilita': viene registrato e mostrato, ma NON filtra il retrieval "
                "(vedi hybrid_retriever.py per il perche').")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()    
    main_logger = setup_main_logger(MAIN_LOGGER_NAME)

    main_logger.info(f"Parsing {args.pdf} ...")
    parsed = parse_pdf(args.pdf, doc_id=args.doc_id)
    images = parsed.images
    attempted_pages = [p for p in parsed.pages if p.ocr_attempted]
    ocr_pages = [p for p in parsed.pages if p.ocr_used]
    empty_pages = [p for p in parsed.pages if p.ocr_empty]
    failed_pages = [p for p in parsed.pages if p.ocr_failed]

    tot_pages = len(parsed.pages)
    tot_images = len(images)
    tot_ocr = len(attempted_pages) # Pagine con testo nativo insufficiente, passate all'OCR
    tot_success = len(ocr_pages)   # OCR riuscito: testo utilizzabile
    tot_empty = len(empty_pages)   # OCR eseguito, ma pagina quasi priva di testo (copertine, separatori)
    tot_failed = len(failed_pages) # OCR non eseguibile: tesseract assente, lingua mancante, pagina corrotta

    main_logger.info(
        f"Completato: {tot_pages} pagine, {tot_images} immagini. "
        f"OCR attivato su {tot_ocr} pagine -> "
        f"Riuscito: {tot_success} | Vuote: {tot_empty} | Fallito: {tot_failed}"
    )

    main_logger.info("Chunking...")
    chunks = chunk_document(parsed, version=args.version, model=args.model, firmware=args.firmware)
    main_logger.info(f"{len(chunks)} chunk generati.")

    main_logger.info("Indicizzazione (embeddings + BM25 + metadati + bounding box immagini)...")
    index_chunks(
        chunks,
        doc_id=args.doc_id,
        version=args.version,
        model=args.model,
        firmware=args.firmware,
        images=images,
        n_pages=len(parsed.pages),
        n_ocr_pages=len(ocr_pages),
        n_ocr_failed_pages=len(failed_pages),
    )
    main_logger.info(f"Completato: {args.doc_id} {args.version}"
          + (f" (modello {args.model})" if args.model else ""))


if __name__ == "__main__":
    main()
