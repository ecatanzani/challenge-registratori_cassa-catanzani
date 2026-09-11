from pathlib import Path

import pandas as pd
import streamlit as st

from src.generation.chain import answer_query
from src.generation.followup import componi_query
from src.ingestion.indexer import index_is_ready
from src.retrieval.hybrid_retriever import retrieve
from src.retrieval.query_processing import normalize_query
from src.retrieval.reranker import rerank

st.set_page_config(page_title="Assistente Manuali Cassa", layout="wide",
                   initial_sidebar_state="collapsed")


def _fmt_bbox(bbox) -> str:
    """Bounding box in forma leggibile, in punti PDF.
    """
    x0, y0, x1, y1 = (round(v) for v in bbox)
    return f"bbox ({x0}, {y0}) – ({x1}, {y1}) pt"


def _etichetta_citazione(c, grassetto: bool = False) -> str:
    """Provenienza di una citazione: documento, versione, pagina, sezione.
    """
    testo = (f"{c.doc_id}"
             + (f" {c.version}" if c.version else "")
             + f" · p.{c.page}"
             + (f" · {c.section_title}" if c.section_title else ""))
    return f"**[{testo}]**" if grassetto else f"📄 {testo}"


st.title("Assistente RAG — Manuali registratore di cassa")
st.caption(
    "Fai una domanda in linguaggio naturale su una procedura, un errore o una configurazione. "
    "Le risposte sono ancorate ai manuali indicizzati, con citazioni e immagini."
)

with st.sidebar:
    st.header("Opzioni")
    show_images = st.toggle("Mostra immagini", value=True)
    show_why_panel = st.toggle("Mostra pannello 'Perche' questa risposta?'", value=True)

if not index_is_ready():
    st.error(
        "**Indice non ancora costruito.** Prima di usare l'assistente serve indicizzare "
        "almeno un manuale:\n\n"
        "```bash\n"
        "python ingest.py --pdf data/manuals/printf_f_manuale_v03.pdf \\\n"
        "    --doc-id printf_f --version v03 --model \"PRINT! F\"\n"
        "```"
    )
    st.stop()


@st.cache_resource(show_spinner="Preparazione dei modelli...")
def _warmup() -> None:
    domanda = normalize_query("come cambio il rotolo della carta")
    rerank(domanda.corrected, retrieve(domanda).chunks)


_warmup()

query = st.text_input(
    "Domanda",
    placeholder="Es. Errore E60: carta esaurita, come risolvo?",
)
submitted = st.button("Chiedi", type="primary")

st.session_state.setdefault("chiarimento_in_sospeso", None)

if submitted and query.strip():
    query_effettiva = componi_query(st.session_state["chiarimento_in_sospeso"], query)
    if query_effettiva != query.strip():
        st.caption(f"↩️ Ho letto la tua risposta come: «{query_effettiva}»")

    with st.spinner("Ricerca nei manuali in corso..."):
        try:
            result = answer_query(query_effettiva)
        except Exception as exc:
            st.error(f"Non sono riuscito a completare la richiesta: {type(exc).__name__}: {exc}")
            st.caption(
                "Se e' la prima esecuzione, controlla che ANTHROPIC_API_KEY sia compilata correttamente "
                "in `.env` e che l'ingestion sia stata eseguita."
            )
            st.stop()

    st.session_state["chiarimento_in_sospeso"] = (
        query_effettiva if result.clarification_question and not result.off_topic else None
    )

    if result.understanding and result.understanding.corrections:
        fixed = ", ".join(f"«{a}» → «{b}»" for a, b in result.understanding.corrections)
        st.caption(f"✏️ Ho interpretato la domanda correggendo: {fixed}")

    if result.off_topic:
        st.info(result.clarification_question)
        if result.related_faqs:
            st.write("**Puoi chiedermi ad esempio:**")
            for faq in result.related_faqs:
                st.markdown(f"- {faq}")
    elif result.ambiguous_models:
        st.warning(result.clarification_question)
        st.write("Modelli trovati tra i risultati:")
        for m in result.ambiguous_models:
            st.markdown(f"- **{m}**")
        st.caption(
            "Suggerimento: ripeti la domanda specificando il modello, es. \"...sul modello X1\"."
        )
        if result.related_faqs:
            st.write("**Oppure una di queste procedure:**")
            for faq in result.related_faqs:
                st.markdown(f"- {faq}")
    elif result.fallback:
        st.warning(
            result.clarification_question
            or "Non ho trovato una procedura sufficientemente pertinente nei manuali indicizzati."
        )
        if result.related_faqs:
            st.write("**FAQ correlate che potrebbero aiutarti:**")
            for faq in result.related_faqs:
                st.markdown(f"- {faq}")
        st.caption(
            "Se nessuna di queste corrisponde, prova a indicare il modello della cassa "
            "o il codice errore che vedi a display."
        )
    else:
        st.subheader("Procedura")
        for step in result.steps:
            st.markdown(f"**{step.number}.** {step.text}")
            citation = next((c for c in result.citations if c.chunk_id == step.chunk_id), None)
            if citation:
                etichetta = _etichetta_citazione(citation)
                if citation.ocr_used:
                    etichetta += "  ·  ⚠️ testo da OCR"
                st.caption(etichetta)

        if result.warnings:
            st.divider()
            for w in result.warnings:
                st.warning(f"⚠️ {w.text}")

        if show_images and result.images:
            st.divider()
            pagine = sorted({img.page for img in result.images})
            n_img = len(result.images)
            st.subheader(
                f"Immagini di riferimento — {n_img} "
                f"{'figura' if n_img == 1 else 'figure'} su "
                f"{len(pagine)} {'pagina' if len(pagine) == 1 else 'pagine'}"
            )
            st.caption(
                "Prima le pagine del manuale con le aree pertinenti evidenziate, poi ogni "
                "figura ingrandita con la sua provenienza. Le figure sono **tutte** quelle "
                "pertinenti delle sezioni citate, non solo la principale."
            )

            anteprime: dict[int, tuple[str, int]] = {}
            for img in result.images:
                if img.page_preview_path and Path(img.page_preview_path).exists():
                    anteprime.setdefault(img.page, (img.page_preview_path, img.highlighted_on_page))

            for numero, (percorso, n_aree) in sorted(anteprime.items()):
                aree = "1 area evidenziata" if n_aree <= 1 else f"{n_aree} aree evidenziate"
                st.caption(f"📄 pagina {numero} — {aree}")
                st.image(percorso, width="stretch")

            if anteprime:
                st.markdown("**Dettaglio delle figure**")
            cols = st.columns(min(n_img, 3))
            for i, img in enumerate(result.images):
                with cols[i % len(cols)]:
                    if img.crop_path and Path(img.crop_path).exists():
                        st.image(img.crop_path, width="stretch",
                                 caption=f"📄 {img.doc_id} · p.{img.page} · {_fmt_bbox(img.bbox)}")
                    else:
                        st.caption(f"(immagine non disponibile: {img.doc_id} · p.{img.page})")

        st.divider()
        st.subheader("Citazioni")
        for c in result.citations:
            st.markdown(f"{_etichetta_citazione(c, grassetto=True)} — _{c.snippet}..._")
            if c.ocr_used:
                st.caption(
                    "⚠️ Il testo di questa pagina proviene da OCR (pagina scansionata): "
                    "verifica sul manuale originale prima di operazioni fiscali."
                )
            if c.page_preview_path and Path(c.page_preview_path).exists():
                with st.expander(f"Apri la pagina {c.page} del manuale"):
                    st.image(c.page_preview_path, width="stretch")

    if show_why_panel:
        with st.expander("🔍 Perche' questa risposta? (interpretazione, frammenti e punteggi)",
                         expanded=False):
            u = result.understanding
            if u:
                st.markdown("**Come ho interpretato la domanda**")
                rows = [("Domanda originale", u.original)]
                if u.corrected != u.original:
                    rows.append(("Domanda usata per la ricerca semantica", u.corrected))
                if u.matched_terms:
                    rows.append(("Termini di dominio riconosciuti", ", ".join(u.matched_terms)))
                if u.error_codes:
                    rows.append(("Codici errore rilevati", ", ".join(u.error_codes)))
                if u.model:
                    rows.append(("Modello indicato", u.model))
                if u.firmware:
                    rows.append(("Firmware indicato", u.firmware))
                st.table(pd.DataFrame(rows, columns=["", "Valore"]).set_index(""))

            if result.retrieved_chunks:
                st.markdown("**Frammenti recuperati e punteggi**")
                df = pd.DataFrame(
                    [
                        {
                            "chunk_id": rc.chunk_id, "doc": rc.doc_id,
                            "versione": rc.version, "pagina": rc.page, "score": rc.score,
                        }
                        for rc in result.retrieved_chunks
                    ]
                )
                
                st.dataframe(
                    df.sort_values("score", ascending=False),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "score": st.column_config.ProgressColumn(
                            "Punteggio di rilevanza",
                            help="Rilevanza stimata dal cross-encoder: quanto il frammento "
                                 "sembra rispondere alla domanda, da 0 a 1",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                    },
                )
            else:
                st.caption("Nessun frammento recuperato.")

            timing = (
                f"Retrieval: {result.retrieval_ms} ms (budget <2000 ms) · "
                f"totale: {result.latency_ms} ms (budget <8000 ms)"
            )
            if result.time_to_first_token_ms is not None:
                timing += f" · primo token a {result.time_to_first_token_ms} ms dalla domanda"
            if result.request_id:
                timing += f" · request_id `{result.request_id}`"
            st.caption(timing)

elif submitted:
    st.info("Scrivi prima una domanda.")
