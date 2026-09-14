"""
Orchestrazione end-to-end: normalizzazione query -> retrieval ibrido ->
rerank -> (eventuale disambiguazione) -> generazione strutturata via LLM ->
validazione grounding (chunk_id devono esistere tra quelli recuperati) ->
arricchimento con crop immagini -> audit log.
"""
import json
import time

from langchain_core.prompts import ChatPromptTemplate

from config import settings
from src.audit.logger import log_query
from src.audit.timing import log_elapsed, new_request_id, timed
from src.generation.faq import faqs_from_chunks, faqs_from_index
from src.generation.prompts import SYSTEM_PROMPT, USER_TEMPLATE, format_context
from src.generation.schemas import (
    Citation, FinalAnswer, RetrievedChunkView, StructuredAnswer,
)
from src.images.crop import crop_images_for_chunk
from src.llm.client import get_chat_model
from src.retrieval.hybrid_retriever import RetrievedChunk, retrieve
from src.retrieval.query_processing import normalize_query
from src.retrieval.reranker import rerank

RETRIEVAL_BUDGET_MS = 2000  # "TTFT retrieval < 2 s"
E2E_BUDGET_MS = 8000        # "E2E < 8 s"


def _generate_structured(
    query: str, chunks: list[RetrievedChunk], t0: float
) -> tuple[StructuredAnswer, float | None]:
    """Ritorna (risposta strutturata, secondi trascorsi DALLA DOMANDA al
    primo frammento prodotto dall'LLM, oppure None se non misurabile).

    L'output finale resta un oggetto Pydantic validato (with_structured_output,
    via tool-calling) — la UI non streamma nulla in tempo reale, per scelta
    esplicita: si preferisce la garanzia sintattica di Pydantic alla
    percezione di rapidita' dello streaming carattere-per-carattere.
    """
    context = format_context(chunks)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_TEMPLATE)]
    )
    llm = get_chat_model()
    structured_llm = llm.with_structured_output(StructuredAnswer)
    chain = prompt | structured_llm

    t_call_start = time.perf_counter()
    first_token_s: float | None = None
    result = None
    try:
        for partial in chain.stream({"context": context, "query": query}):
            if first_token_s is None:
                # Dalla domanda dell'utente, non dall'inizio della chiamata LLM.
                first_token_s = time.perf_counter() - t0
                log_elapsed("generazione.primo_frammento_llm (dalla domanda)", first_token_s)
            result = partial  # l'ultimo frammento e' l'oggetto completo
        if result is None:
            raise RuntimeError("stream vuoto")
    except Exception as exc:
        if settings.debug_timing_logs:
            print(f"[TIMING] generazione.stream_non_disponibile ({exc!r}), fallback a invoke()")
        result = chain.invoke({"context": context, "query": query})
        first_token_s = None

    log_elapsed("generazione.totale_llm", time.perf_counter() - t_call_start)
    return result, first_token_s


def _validate_grounding(
    answer: StructuredAnswer, chunks: list[RetrievedChunk]
) -> StructuredAnswer:
    """Rimuove qualsiasi step/warning che citi un chunk_id non realmente
    presente tra i chunk recuperati: guardrail anti-allucinazione
    strutturale"""
    valid_ids = {c.document.metadata.get("chunk_id") for c in chunks}
    kept_steps = [s for s in answer.steps if s.chunk_id in valid_ids]

    if answer.steps and not kept_steps:
        # Il modello ha citato chunk_id inesistenti
        return StructuredAnswer(
            fallback=True,
            related_faqs=answer.related_faqs or faqs_from_chunks(chunks),
            clarification_question="Non sono riuscito ad ancorare la risposta ai documenti disponibili.",
        )

    kept_warnings = [w for w in answer.warnings if w.chunk_id in valid_ids or w.chunk_id is None]
    answer.steps = kept_steps
    answer.warnings = kept_warnings
    return answer


def _chunk_views(chunks: list[RetrievedChunk]) -> list[RetrievedChunkView]:
    return [
        RetrievedChunkView(
            chunk_id=c.document.metadata.get("chunk_id"),
            doc_id=c.document.metadata.get("doc_id"),
            page=c.document.metadata.get("page_start"),
            score=c.score,
            snippet=c.document.page_content[:200],
            version=c.document.metadata.get("version"),
        )
        for c in chunks
    ]


def _build_citations_and_images(
    structured: StructuredAnswer, chunks: list[RetrievedChunk]
) -> tuple[list[Citation], list]:
    """Citazioni per ogni chunk effettivamente citato dai passi, piu' TUTTE le
    immagini pertinenti della SEZIONE citata
    """
    chunk_by_id = {c.document.metadata.get("chunk_id"): c for c in chunks}
    citations: list[Citation] = []
    images = []
    seen_image_ids: set[str] = set()
    preview_by_page: dict[tuple[str, int], str | None] = {}
    gruppi_elaborati: set[tuple[str, frozenset[str]]] = set()
    cited_ids: list[str] = []
    for chunk_id in (
        [s.chunk_id for s in structured.steps]
        + [w.chunk_id for w in structured.warnings if w.chunk_id]
    ):
        if chunk_id not in cited_ids:
            cited_ids.append(chunk_id)

    # Estrazione dei metadati per ongi chunk
    for chunk_id in cited_ids:
        c = chunk_by_id.get(chunk_id)
        if not c:
            continue
        meta = c.document.metadata
        doc_id = meta.get("doc_id")
        page = meta.get("page_start")

        try:
            image_ids = json.loads(meta.get("image_ids", "[]"))
        except (TypeError, json.JSONDecodeError):
            image_ids = []

        gruppo = (doc_id, frozenset(image_ids))
        if image_ids and gruppo not in gruppi_elaborati:
            gruppi_elaborati.add(gruppo)
            for img in crop_images_for_chunk(doc_id, page, image_ids):
                preview_by_page.setdefault((doc_id, img.page), img.page_preview_path)
                if img.image_id in seen_image_ids:
                    continue
                if _images_capped(images):
                    break
                seen_image_ids.add(img.image_id)
                images.append(img)

        citations.append(
            Citation(
                doc_id=doc_id,
                page=page,
                section_title=meta.get("section_title") or None,
                chunk_id=chunk_id,
                snippet=c.document.page_content[:220],
                version=meta.get("version"),
                ocr_used=bool(meta.get("ocr_used")),
                page_preview_path=preview_by_page.get((doc_id, page)),
            )
        )

    return citations, images


def _images_capped(images: list) -> bool:
    """La valvola di sicurezza, disattivabile con MAX_IMAGES_PER_ANSWER=0."""
    limit = settings.max_images_per_answer
    return limit > 0 and len(images) >= limit


def answer_query(raw_query: str) -> FinalAnswer:
    request_id = new_request_id()
    t0 = time.perf_counter()

    with timed("query.normalizzazione"):
        normalized = normalize_query(raw_query)

    with timed("retrieval.totale"):
        retrieval = retrieve(normalized)

    if retrieval.off_topic:
        # Nessuna chiamata LLM: la domanda e' fuori dal dominio dei manuali indicizzati
        retrieval_ms = int((time.perf_counter() - t0) * 1000)
        latency_ms = retrieval_ms
        final = FinalAnswer(
            query=raw_query,
            steps=[], citations=[], images=[], warnings=[],
            fallback=True,
            off_topic=True,
            related_faqs=faqs_from_index(limit=3),
            clarification_question=(
                "Questa domanda non sembra riguardare i manuali del registratore di cassa "
                "indicizzati. Posso aiutarti solo con procedure, errori e configurazioni "
                "descritte nei manuali — prova a riformulare, oppure guarda le FAQ correlate."
            ),
            ambiguous_models=[],
            retrieved_chunks=[],
            understanding=normalized,
            request_id=request_id,
            time_to_first_token_ms=None,
            retrieval_ms=retrieval_ms,
            latency_ms=latency_ms,
        )
        log_query(raw_query, final)
        return final

    with timed("rerank.totale"):
        chunks = rerank(normalized.corrected, retrieval.chunks)

    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    if retrieval.ambiguous:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        final = FinalAnswer(
            query=raw_query,
            steps=[], citations=[], images=[], warnings=[],
            fallback=True,
            related_faqs=faqs_from_chunks(chunks, limit=3),
            clarification_question=(
                "Ho trovato procedure simili per piu' modelli diversi: "
                f"{', '.join(retrieval.candidate_models)}. Puoi specificare il modello o il firmware "
                "del registratore di cassa?"
            ),
            ambiguous_models=retrieval.candidate_models,
            retrieved_chunks=_chunk_views(chunks),
            understanding=normalized,
            request_id=request_id,
            time_to_first_token_ms=None,
            retrieval_ms=retrieval_ms,
            latency_ms=latency_ms,
        )
        log_query(raw_query, final)
        return final

    structured, first_token_s = _generate_structured(raw_query, chunks, t0)

    with timed("generazione.validazione_grounding"):
        structured = _validate_grounding(structured, chunks)

    with timed("immagini.ritaglio_totale"):
        citations, images = _build_citations_and_images(structured, chunks)
    
    related_faqs = structured.related_faqs
    if structured.fallback and not related_faqs:
        related_faqs = faqs_from_chunks(chunks, limit=3)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    time_to_first_token_ms = int(first_token_s * 1000) if first_token_s is not None else None
    
    final = FinalAnswer(
        query=raw_query,
        steps=structured.steps,
        citations=citations,
        images=images,
        warnings=structured.warnings,
        fallback=structured.fallback,
        off_topic=False,
        related_faqs=related_faqs,
        clarification_question=structured.clarification_question,
        ambiguous_models=[],
        retrieved_chunks=_chunk_views(chunks),
        understanding=normalized,
        request_id=request_id,
        time_to_first_token_ms=time_to_first_token_ms,
        retrieval_ms=retrieval_ms,
        latency_ms=latency_ms,
    )
    log_query(raw_query, final)
    return final
