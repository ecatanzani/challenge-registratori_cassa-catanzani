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
python ingest.py --pdf data/manuals/printf_f_manuale_v03.pdf --doc-id stampante_x1 --version 1.2 --model "X1" --firmware "2.3.1"
```

| Argomento | Ruolo |
|---|---|
| `--pdf` | percorso del manuale da indicizzare |
| `--doc-id` | identificativo stabile del documento, invariante fra le versioni |
| `--version` | versione del manuale; a parità di `doc-id` sostituisce i chunk della precedente invece di duplicarli |
| `--model` | modello di registratore di cassa a cui il manuale si riferisce |
| `--firmware` | versione firmware documentata |

`--model` e `--firmware` non sono decorativi: finiscono nei metadati di ogni
chunk e servono a distinguere manuali di apparecchi diversi in fase di
retrieval.

Alla **prima** esecuzione viene scaricato il modello di embedding
(`intfloat/multilingual-e5-large`, circa 2 GB) nella cache di Hugging Face in
`~/.cache/huggingface`. Le esecuzioni successive girano offline.

Output atteso:

```
Completato: 73 pagine, 17 immagini. OCR attivato su 1 pagine -> Riuscito: 0 | Vuote: 1 | Fallito: 0
Chunking...
195 chunk generati.
Indicizzazione (embeddings + BM25 + metadati + bounding box immagini)...
Completato: stampante_x1 v1.2 (modello X1)
```

L'indice viene scritto in `data/vectorstore/`: Chroma per i vettori, un JSONL
con il corpus dei chunk per BM25, SQLite per versioning dei manuali e bounding
box delle immagini. Per ripartire da zero e' sufficiente cancellare quella
cartella e rilanciare il comando.

## Strumenti di analisi

`scripts/` contiene strumenti di indagine, non codice di produzione: non
partecipano al funzionamento del sistema, ma le loro dipendenze sono incluse in
`requirements.txt` insieme a tutto il resto, cosi' l'ambiente e il punto di
installazione restano uno solo.

```bash
python scripts/compare_chunking.py --pages 30 31
```

[compare_chunking.py](scripts/compare_chunking.py) mette a confronto, sulle
stesse pagine e con lo stesso modello di embedding, il chunking strutturato
adottato dal progetto e il `SemanticChunker` di LangChain, riportando per
entrambi numero di chunk, lunghezze e quota di blocchi con un titolo di
sezione riconosciuto.
