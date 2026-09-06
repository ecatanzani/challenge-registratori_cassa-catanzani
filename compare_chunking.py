"""
Confronta il chunking strutturato del progetto (regex sulle intestazioni) con
SemanticChunker di LangChain, sulle stesse pagine e con lo stesso modello di
embedding.
"""

import argparse
import re
import sys
from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker

from src.ingestion.chunker import chunk_document
from src.ingestion.pdf_parser import ParsedDocument, parse_pdf
from src.llm.client import get_embedding_model


def _show_chunks(titolo: str, blocchi: list[tuple[str | None, str]]) -> None:
    print(f"\n{'=' * 92}\n  {titolo} — {len(blocchi)} chunk\n{'=' * 92}")
    for i, (sezione, testo) in enumerate(blocchi, 1):
        testo = re.sub(r"\s+", " ", testo).strip()
        print(f"\n[{i}] {len(testo):>4} car.   sezione: {sezione or '(nessuna)'}")
        print(f"    {testo[:260]}{'...' if len(testo) > 260 else ''}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Chunking Method Comparison")
    ap.add_argument("--pdf", default="data/manuals/printf_f_manuale_v03.pdf")
    ap.add_argument("--pages", type=int, nargs="+", required=True, help="pagine PDF, 1-indexed")
    ap.add_argument("--percentile", type=int, default=95, help="soglia di taglio del semantico")
    args = ap.parse_args()
    return args

def main() -> None:
    args = parse_args()

    parsed = parse_pdf(args.pdf, doc_id="cmp")
    pagine = [p for p in parsed.pages if p.page in set(args.pages)]
    if not pagine:
        sys.exit(f"Pagine {args.pages} non trovate in {args.pdf}")

    strutturati = [
        (c.section_title, c.text)
        for c in chunk_document(ParsedDocument("cmp", args.pdf, pagine), version="cmp")
    ]

    testo = "\n".join(p.text for p in pagine)
    semantico = SemanticChunker(
        get_embedding_model(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=args.percentile,
    )
    semantici = [(None, c) for c in semantico.split_text(testo)]

    print(f"Pagine {sorted(args.pages)} — {len(testo)} caratteri estratti.")
    _show_chunks("STRUTTURATO (regex sulle intestazioni)", strutturati)
    _show_chunks(f"SEMANTICO (percentile {args.percentile})", semantici)

    def mediana(blocchi):
        return sorted(len(t) for _, t in blocchi)[len(blocchi) // 2]

    print(f"\n{'-' * 92}")
    print(f"{'':14}{'chunk':>7}{'mediana':>10}{'min':>7}{'max':>7}{'con titolo':>13}")
    for nome, b in (("strutturato", strutturati), ("semantico", semantici)):
        lung = [len(t) for _, t in b]
        titolati = sum(1 for s, _ in b if s)
        print(f"{nome:14}{len(b):>7}{mediana(b):>10}{min(lung):>7}{max(lung):>7}"
              f"{f'{titolati}/{len(b)}':>13}")


if __name__ == "__main__":
    main()
