"""
Schema strutturato della risposta. Il punto chiave anti-allucinazione: ogni
StepItem porta un `chunk_id` obbligatorio che DEVE corrispondere a uno dei
chunk realmente recuperati (validato in chain.py, non solo richiesto via
prompt). Se il modello non riesce ad ancorare uno step a un chunk esistente,
quello step viene scartato in post-processing e, se non resta nulla di
valido, scatta il fallback.
"""
from __future__ import annotations

import dataclasses

from pydantic import BaseModel, Field, field_serializer

from src.retrieval.query_processing import NormalizedQuery


class StepItem(BaseModel):
    number: int = Field(description="Numero progressivo del passo, a partire da 1")
    text: str = Field(description="Testo del passo, linguaggio semplice e diretto")
    chunk_id: str = Field(description="ID del chunk del manuale da cui questo passo e' tratto")


class AnswerWarning(BaseModel):
    """Avvertenza o nota di conformita'.

    Si chiama AnswerWarning e non Warning per non ombreggiare l'eccezione
    built-in `Warning` all'interno dei moduli che importano questo schema.
    """

    text: str = Field(
        description="Avvertenza di ruolo o di conformita' fiscale, in UNA frase. Non usare "
                    "per prerequisiti, tempi di attesa o risoluzione problemi: quelli "
                    "appartengono al passo della procedura."
    )
    chunk_id: str | None = Field(default=None, description="Chunk di riferimento se applicabile")


class StructuredAnswer(BaseModel):
    steps: list[StepItem] = Field(default_factory=list)
    warnings: list[AnswerWarning] = Field(default_factory=list)
    fallback: bool = Field(
        default=False,
        description="True se non e' stato possibile costruire una risposta ancorata ai documenti",
    )
    related_faqs: list[str] = Field(
        default_factory=list, description="FAQ correlate proposte in caso di fallback"
    )
    clarification_question: str | None = Field(
        default=None, description="Domanda di chiarimento strutturata in caso di ambiguita'"
    )


class ImageCitation(BaseModel):
    image_id: str
    doc_id: str
    page: int
    bbox: tuple[float, float, float, float] = Field(
        description="Area della figura nella pagina PDF (x0, y0, x1, y1), in punti"
    )
    crop_path: str | None = Field(default=None, description="Ritaglio della sola figura")
    page_preview_path: str | None = Field(
        default=None,
        description="Pagina intera del manuale con TUTTE le aree pertinenti evidenziate da un "
                    "riquadro. E' la destinazione del link 'apri la pagina originale', ed e' "
                    "condivisa da tutte le figure della stessa pagina.",
    )
    highlighted_on_page: int = Field(
        default=1,
        description="Quante aree sono evidenziate nell'anteprima di pagina. Serve alla UI per "
                    "dirlo all'operatore ('2 aree evidenziate') invece di lasciargli scoprire "
                    "da solo che il secondo riquadro esiste.",
    )


class Citation(BaseModel):
    doc_id: str
    page: int
    section_title: str | None
    chunk_id: str
    snippet: str
    version: str | None = Field(default=None, description="Versione del manuale citato")
    ocr_used: bool = Field(
        default=False,
        description="True se il testo di questa pagina proviene da OCR e non dal livello testo "
                    "nativo del PDF: l'informazione resta valida ma e' intrinsecamente meno "
                    "affidabile, e va mostrata all'operatore.",
    )
    page_preview_path: str | None = Field(
        default=None, description="Anteprima della pagina citata, per il link alla pagina originale"
    )


class RetrievedChunkView(BaseModel):
    """Vista semplificata di un chunk recuperato, per il pannello 'Perche' questa risposta?'."""
    chunk_id: str
    doc_id: str
    page: int
    score: float
    snippet: str
    version: str | None = None


class FinalAnswer(BaseModel):
    query: str
    steps: list[StepItem]
    citations: list[Citation]
    images: list[ImageCitation]
    warnings: list[AnswerWarning]
    fallback: bool
    off_topic: bool = Field(
        default=False,
        description="True se la domanda e' stata giudicata fuori dal dominio dei manuali prima ancora di chiamare l'LLM",
    )
    related_faqs: list[str]
    clarification_question: str | None
    ambiguous_models: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkView] = Field(default_factory=list)
    understanding: NormalizedQuery | None = Field(
        default=None,
        description="Cosa il sistema ha capito della domanda prima di cercare: refusi "
                    "corretti, sinonimi riconosciuti, codici errore, modello. Alimenta il "
                    "pannello 'Perche' questa risposta?' — se una risposta e' fuori "
                    "bersaglio, molto spesso il motivo e' qui e non nel retrieval.",
    )
    request_id: str | None = Field(
        default=None,
        description="Identificativo breve della richiesta, per correlare le righe [TIMING] "
                    "in stdout e la voce corrispondente nell'audit log — vedi src/audit/timing.py",
    )
    time_to_first_token_ms: int | None = Field(
        default=None,
        description="Tempo dalla DOMANDA dell'utente al primo frammento prodotto dall'LLM, in ms "
                    "(include normalizzazione, retrieval e rerank: e' il tempo che l'operatore "
                    "aspetta davvero). None se non misurabile.",
    )
    retrieval_ms: int = Field(
        default=0,
        description="Tempo di solo retrieval (normalizzazione + ricerca ibrida + rerank), in ms. "
                    "E' la metrica confrontata con il budget 'TTFT retrieval < 2s' del brief.",
    )
    latency_ms: int = 0

    @field_serializer("understanding")
    def _understanding_senza_expanded(self, valore: NormalizedQuery | None):
        """Serializza la NormalizedQuery escludendo `expanded`.

        `expanded` e' la query con tutti i sinonimi appesi, costruita per
        BM25: un artefatto interno che su domande con piu' famiglie
        riconosciute supera i 500 caratteri. Farlo uscire significherebbe
        scriverlo nell'audit log ad ogni domanda e serializzarlo verso una UI
        che non lo usa. Tutto il resto della NormalizedQuery e' invece
        esattamente cio' che serve al pannello di trasparenza, quindi si
        incapsula la dataclass e si toglie un campo — invece di mantenere un
        secondo modello quasi identico solo per ometterlo.
        """
        if valore is None:
            return None
        return {k: v for k, v in dataclasses.asdict(valore).items() if k != "expanded"}
