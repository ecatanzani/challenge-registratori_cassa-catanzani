# challenge-registratori_cassa-catanzani

Challenge per posizione AI Scientist - Sistema RAG per registratori di cassa

## Setup

Creazione dell'ambiente e installazione delle dipendenze: [INSTALL.md](INSTALL.md).

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

## Interrogazione

```bash
streamlit run app.py
```

Il sistema risponde **a turno singolo**: ogni domanda viene interpretata da
sola, senza storico della conversazione. La scelta e' deliberata — il prompt
resta corto e verificabile, e la risposta e' sempre riconducibile ai soli chunk
citati — ma ha una conseguenza da gestire, perche' il modello e' istruito a
chiedere un chiarimento quando il contesto non basta ("Quale codice errore
compare a display?").

Senza accorgimenti, la risposta dell'operatore rientrerebbe come domanda nuova
e isolata: misurato su questo indice, un `E43` da solo ottiene 0,781 di
similarita' contro una soglia di dominio di 0,797, e verrebbe respinto come
fuori tema — il sistema rifiuterebbe la risposta alla propria domanda.

Per questo l'app conserva **una sola** informazione fra un turno e l'altro: la
domanda che ha generato il chiarimento, che viene anteposta alla risposta
quando questa e' breve e non e' a sua volta una domanda (vedi
[src/generation/followup.py](src/generation/followup.py)). La query cosi'
composta viene mostrata all'operatore, e le domande complete passano sempre
invariate.

## Approfondimenti

- [La soglia del gate di dominio](docs/soglia-gate-dominio.md) — perche' il gate
  esiste, perche' i suoi due errori non costano uguale, e come si ricava il
  numero in `OFF_TOPIC_SIMILARITY_THRESHOLD`.
- [Normalizzazione della query](docs/normalizzazione-query.md) — come la domanda
  viene ripulita prima del retrieval: estrazione dei metadati, correzione dei
  refusi, espansione sinonimica, e perche' la correzione fuzzy ha bisogno di una
  guardia sulle parole funzionali.

## Strumenti di analisi

`scripts/` contiene strumenti di indagine, non codice di produzione: non
partecipano al funzionamento del sistema, ma le loro dipendenze sono incluse in
`requirements.txt` insieme a tutto il resto, cosi' l'ambiente e il punto di
installazione restano uno solo.

```bash
python scripts/compare_chunking.py --pages 30 31
```

[calibrate_threshold.py](scripts/calibrate_threshold.py) ricalcola la soglia del
gate di dominio (`OFF_TOPIC_SIMILARITY_THRESHOLD`) sui punteggi reali
dell'indice. Va rilanciato dopo ogni re-ingestione, per accorgersi se i punteggi
si sono spostati; il criterio dietro la formula e' in
[docs/soglia-gate-dominio.md](docs/soglia-gate-dominio.md).

[compare_chunking.py](scripts/compare_chunking.py) mette a confronto, sulle
stesse pagine e con lo stesso modello di embedding, il chunking strutturato
adottato dal progetto e il `SemanticChunker` di LangChain, riportando per
entrambi numero di chunk, lunghezze e quota di blocchi con un titolo di
sezione riconosciuto.
