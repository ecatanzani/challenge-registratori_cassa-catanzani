# challenge-registratori_cassa-catanzani

Challenge per la realizzazione di un sistema RAG per registratori di cassa.

Il sistema permette agli utenti di effettuare delle domande, generiche o specifiche, per ottenere supporto in merito all'utilizzo di registratori di cassa.

Il sistema include il codice per servire una pagina web, attraverso la quale l'utente interagirà con il sistema.

## Descrizione generale del sistema

Il sistema realizzato può essere sintetizzato con il seguente schema:

![Schema dei componenti: ingestion, indice, inferenza e serving](docs/architettura.svg)

#### Ingestion dei dati

- Parsing del PDF (testo ed immagini)
- Recupero pagine scansionate con sistema OCR
- Chunking del testo
- Embedding del testo
- Vector Store (Chroma DB)
- Corpus di BM25 (JSONL)
- Database SQLite per versioni, immagini e bounding box

#### Inferenza

- Normalizzazione della query (correzione errori - ricerca termini simili)
- Ricerca densa
- Gate di dominio (per reiezione domande off-topic)
- Ricerca semantica
- Fusione risultati retriever semantico e BM25 con tecnica RRF
- Reranker
- LLM per la generazione della risposta
- Validazione della risposta
- Recupero delle immagini (se presenti) nei chunk citati

#### Serving

- Serving pagina web tramite framework `Streamlit`

#### Logging

- Logging delle metriche di performance su disco (file JSONL)

## Setup

Creazione dell'ambiente e installazione delle dipendenze: [INSTALL.md](INSTALL.md).
In alternativa, senza installare nulla oltre a Docker: [Avvio del sistema con Docker](#avvio-del-sistema-con-docker).

## Configurazione

Le impostazioni si leggono da un file `.env` nella radice del repository.
`.env.example` elenca le variabili disponibili con i rispettivi default:

```bash
cp .env.example .env
```

Per l'ingestion **non serve nessuna chiave**: gli embedding sono calcolati in
locale. `ANTHROPIC_API_KEY` diventa necessaria solo per la fase di risposta,
che usa l'API Anthropic.

Ogni campo della classe `Settings` in [config.py](config.py) e' sovrascrivibile
dalla variabile d'ambiente omonima in maiuscolo — per esempio `EMBEDDING_MODEL`
o `TOP_K_FINAL`.

## Ingestion

L'ingestion trasforma un manuale PDF nell'indice interrogabile: estrae testo e
immagini pagina per pagina (con OCR di fallback sulle pagine scansionate), lo
divide in chunk e ne persiste embedding, corpus lessicale e metadati.

```bash
python ingest.py --pdf data/manuals/printf_f_manuale_v03.pdf --doc-id printf_f --version v03 --model "PRINT! F"
```

| Argomento | Ruolo |
|---|---|
| `--pdf` | percorso del manuale da indicizzare |
| `--doc-id` | identificativo stabile del documento, invariante fra le versioni |
| `--version` | versione del manuale; a parità di `doc-id` sostituisce i chunk della precedente invece di duplicarli |
| `--model` | modello di registratore di cassa a cui il manuale si riferisce |
| `--firmware` | versione firmware documentata (opzionale: questo manuale non ne dichiara una) |

`--model` e `--firmware` non sono decorativi: finiscono nei metadati di ogni
chunk e servono a distinguere manuali di apparecchi diversi in fase di
retrieval.

`--doc-id` e `--version` del comando qui sopra non sono d'esempio: sono i valori
che l'eval set in `eval/eval_set.jsonl` si aspetta di trovare nell'indice.
Indicizzare con un `--doc-id` diverso non produce nessun errore visibile, ma fa
riportare a `scripts/eval.py` uno hit@5 di 0/24 su un sistema che funziona.

Alla **prima** esecuzione viene scaricato il modello di embedding
(`intfloat/multilingual-e5-large`, circa 2 GB) nella cache di Hugging Face in
`~/.cache/huggingface`. Le esecuzioni successive girano offline.

Output atteso:

```
Completato: 73 pagine, 17 immagini. OCR attivato su 1 pagine -> Riuscito: 0 | Vuote: 1 | Fallito: 0
Chunking...
195 chunk generati.
Indicizzazione (embeddings + BM25 + metadati + bounding box immagini)...
Completato: printf_f v03 (modello PRINT! F)
```

L'indice viene scritto in `data/vectorstore/`: Chroma per i vettori, un JSONL
con il corpus dei chunk per BM25, SQLite per versioning dei manuali e bounding
box delle immagini. Per ripartire da zero e' sufficiente cancellare quella
cartella e rilanciare il comando.

Il sistema risponde **a turno singolo**: ogni domanda viene interpretata da
sola, senza storico della conversazione. La scelta e' deliberata — il prompt
resta corto e verificabile, e la risposta e' sempre riconducibile ai soli chunk
citati. Il sistema è istruito a chiedere un chiarimento quando il contesto non basta.

## Avvio del sistema da ambiente python locale

Prima di avviare il sistema è necessario effettuare l'ingestion dei dati, tramite il seguente comando:

```python
python ingest.py --pdf data/manuals/printf_f_manuale_v03.pdf --doc-id printf_f --version v03 --model "PRINT! F"
```

Al termine del processo è possibile avviare il server web tramite il seguente comando:

```bash
streamlit run app.py --server.fileWatcherType none
```

Il flag spegne il file watcher di Streamlit, che serve solo a ricaricare l'app
quando si modifica il codice.

## Avvio del sistema con Docker

Prima di avviare il sistema è necessario effettuare l'ingestion dei dati, tramite il seguente comando:

```bash
docker compose run --rm app python ingest.py --pdf data/manuals/printf_f_manuale_v03.pdf --doc-id printf_f --version v03 --model "PRINT! F"
```

Al termine del processo è possibile avviare il server web tramite il seguente comando:

```bash
docker compose up
```

## Eval

Le prestazioni del sistema si misurano con lo script di valutazione end-to-end,
che interroga il sistema completo su 40 domande etichettate in
[eval/eval_set.jsonl](eval/eval_set.jsonl):

```bash
python scripts/eval.py
```

Risultati di riferimento sul codice attuale:

| metrica | valore |
|---|---|
| hit@5 (pagina giusta fra i 5 frammenti passati all'LLM) | 24/24 |
| decisioni corrette (risposta, fallback, rifiuto) | 39/40 |
| figure attese allegate | 12/12 |
| retrieval, media a regime | 477 ms |
| risposta completa, media a regime | 4832 ms |

Metriche per classe, tempi per fase e per tipo di risposta, robustezza ai refusi
e condizioni della misura sono in [docs/valutazione-prestazioni.md](docs/valutazione-prestazioni.md).

## Approfondimenti

- [Normalizzazione della query](docs/normalizzazione-query.md) — questo documento spiega come la domanda dell'utente viene ripulita prima del retrieval: estrazione dei metadati, correzione dei refusi sulle parole del manuale indicizzato, riconoscimento dei termini per l'espansione sinonimica, e le misure con cui i due stadi sono stati tarati.
- [La soglia del gate di dominio](docs/soglia-gate-dominio.md) — questo documento spiega che cosa si intende con gate di dominio e come si ricava il numero in `OFF_TOPIC_SIMILARITY_THRESHOLD`.
- [Fusione RRF e reranker](docs/rrf-e-reranker.md) — questo documento argomenta la logica dietro il sistema di reranking e perche' il gruppo di candidati resta a 5 (come indicato nel file di configurazione).

## Strumenti di analisi

La cartella `scripts` contiene strumenti di indagine, non codice di produzione: non partecipano al funzionamento del sistema, ma le loro dipendenze sono incluse in `requirements.txt`.

[calibrate_threshold.py](scripts/calibrate_threshold.py) calcola la soglia del gate di dominio (`OFF_TOPIC_SIMILARITY_THRESHOLD`). Va rilanciato dopo ogni re-ingestione, per accorgersi se i punteggi
si sono spostati; il criterio dietro la formula è spiegato dettagliatamente in [docs/soglia-gate-dominio.md](docs/soglia-gate-dominio.md).

[calibrate_fuzzy.py](scripts/calibrate_fuzzy.py) misura, al variare di
`FUZZY_THRESHOLD`, quanti refusi sintetici vengono ricondotti al termine di dominio giusto e quanto testo corretto del manuale viene riscritto.

[compare_chunking.py](scripts/compare_chunking.py) mette a confronto, sulle stesse pagine e con lo stesso modello di embedding, il chunking strutturato adottato dal progetto e il `SemanticChunker` di LangChain, riportando per entrambi numero di chunk, lunghezze e quota di blocchi con un titolo di sezione riconosciuto.
