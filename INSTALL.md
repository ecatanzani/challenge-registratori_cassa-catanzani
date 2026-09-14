# Installazione

Istruzioni per creare un ambiente pulito ed eseguire l'applicativo.

Le opzioni di installazione sono due: creazione di un ambiente Python locale (sezioni 1–4) oppure usare Docker (sezione 5).

## 1. Prerequisiti

**Python 3.11 o superiore.**

```bash
python3 --version
```

**Tesseract**, usato per l'OCR di fallback sulle pagine scansionate.

```bash
brew install tesseract tesseract-lang            # macOS
sudo apt install tesseract-ocr tesseract-ocr-ita # Debian/Ubuntu
```

Tesseract non è un componente fondamentale; in sua assenza, ed in caso di pagine scansionate nel manuale, il metodo di fallback OCR fallirà.

## 2. Creazione dell'ambiente virtuale

Dalla radice del repository:

```bash
python3 -m venv .venv
```

Attivazione (da ripetere a ogni nuova shell):

```bash
source .venv/bin/activate
```

A prompt attivato, `python` e `pip` puntano all'ambiente e non
all'installazione di sistema. Verifica:

```bash
which python
```

## 3. Aggiornamento di pip

`venv` installa una versione di pip che può essere più vecchia di quella
necessaria a risolvere alcune wheel:

```bash
python -m pip install --upgrade pip
```

## 4. Installazione delle dipendenze

```bash
pip install -r requirements.txt
```

A installazione completata, configurazione ed esecuzione sono descritte nel
[README](README.md).

## 5. Alternativa: esecuzione con Docker

L'immagine contiene Python, le dipendenze e Tesseract con la lingua italiana.
Il codice viene copiato dentro l'immagine; la cartella `data/` del repository
(manuali, indice, audit log) resta invece sull'host ed è montata nel container.

### Prerequisiti

**Docker con il plugin Compose**: Docker Desktop su macOS e Windows, Docker
Engine con `docker-compose-plugin` su Linux.

```bash
docker compose version
```

**Almeno 4 GB di memoria assegnata a Docker** (Docker Desktop → Settings →
Resources). Misurato: picco di 2,6 GB, sia durante l'ingestion sia rispondendo
alle domande.

**Circa 6 GB di spazio su disco**: 3,1 GB per l'immagine, 2,6 GB per i
modelli.

### Configurazione

Compose legge il file `.env` all'avvio di ogni container e senza si ferma con
l'errore `env file .env not found`. Va creato anche solo per l'ingestion, che
però non usa la chiave API:

```bash
cp .env.example .env
```

I percorsi (`MANUALS_DIR`, `VECTORSTORE_DIR`, `CROPS_DIR`, `AUDIT_LOG_PATH`)
vanno lasciati relativi come in `.env.example`: nel container `./data`
corrisponde alla cartella `data/` del repository.

### Costruzione dell'immagine

```bash
docker compose build
```

Di seguito l'output di riferimento:

```bash
❯ docker compose build
[+] Building 22.6s (17/17) FINISHED
 => [internal] load local bake definitions                                                                              0.0s
 => => reading from stdin 634B                                                                                          0.0s
 => [internal] load build definition from Dockerfile                                                                    0.0s
 => => transferring dockerfile: 643B                                                                                    0.0s
 => [internal] load metadata for docker.io/library/python:3.12-slim                                                     0.0s
 => [internal] load .dockerignore                                                                                       0.0s
 => => transferring context: 2B                                                                                         0.0s
 => [ 1/10] FROM docker.io/library/python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e  0.0s
 => => resolve docker.io/library/python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e18  0.0s
 => [internal] load build context                                                                                       0.0s
 => => transferring context: 5.00kB                                                                                     0.0s
 => CACHED [ 2/10] RUN apt-get update     && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-it  0.0s
 => CACHED [ 3/10] WORKDIR /app                                                                                         0.0s
 => CACHED [ 4/10] RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu                0.0s
 => CACHED [ 5/10] COPY requirements.txt .                                                                              0.0s
 => CACHED [ 6/10] RUN pip install --no-cache-dir -r requirements.txt                                                   0.0s
 => CACHED [ 7/10] COPY config.py app.py ingest.py ./                                                                   0.0s
 => CACHED [ 8/10] COPY src ./src                                                                                       0.0s
 => CACHED [ 9/10] COPY scripts ./scripts                                                                               0.0s
 => CACHED [10/10] COPY eval ./eval                                                                                     0.0s
 => exporting to image                                                                                                 22.0s
 => => exporting layers                                                                                                 0.0s
 => => exporting manifest sha256:dd11d83af72bf04c0fae5cc4de007f4dca9a9e993fb7f08a7c05b5d17f169fa3                       0.0s
 => => exporting config sha256:85b06a792d50fe8dd447f3cd531c49106f7f6961f9e8c6450302615bd69d277b                         0.0s
 => => exporting attestation manifest sha256:4f993390d5797a0bcdb7381b489cb951aca522cc1198a2b0e9489b3dca11771c           0.0s
 => => exporting manifest list sha256:2c7df287d2ed9af224007e4e988097cf4aee5a73fc4e4017eb9ec4a0c5365345                  0.0s
 => => naming to docker.io/library/challenge-registratori_cassa-catanzani-app:latest                                    0.0s
 => => unpacking to docker.io/library/challenge-registratori_cassa-catanzani-app:latest                                21.9s
 => resolving provenance for metadata file                                                                              0.0s
[+] build 1/1
 ✔ Image challenge-registratori_cassa-catanzani-app Built
```

La prima volta scarica e installa le dipendenze: circa 2-3 minuti. Dopo ogni modifica al codice l'immagine va ricostruita, perché il codice non è montato ma copiato.

### Ingestion

Il manuale deve trovarsi in `data/manuals/` sul computer host.

```bash
docker compose run --rm app python ingest.py --pdf data/manuals/printf_f_manuale_v03.pdf --doc-id printf_f --version v03 --model "PRINT! F"
```

Di seguito l'output di riferimento:

```bash
docker compose run --rm app python ingest.py --pdf data/manuals/printf_f_manuale_v03.pdf --doc-id printf_f --version v03 --model "PRINT! F"
[+] run 1/1
 ✔ Network challenge-registratori_cassa-catanzani_default Created                                                        0.1s
Container challenge-registratori_cassa-catanzani-app-run-c4720b9b14af Creating
Container challenge-registratori_cassa-catanzani-app-run-c4720b9b14af Created
2026-09-11 20:28:26,932 - ingest - INFO - Logger 'ingest' initialized at level INFO
2026-09-11 20:28:26,932 - ingest - INFO - Parsing data/manuals/printf_f_manuale_v03.pdf ...
2026-09-11 20:28:30,097 - ingest - INFO - Completato: 73 pagine, 17 immagini. OCR attivato su 1 pagine -> Riuscito: 0 | Vuote: 1 | Fallito: 0
2026-09-11 20:28:30,097 - ingest - INFO - Chunking...
2026-09-11 20:28:30,109 - ingest - INFO - 196 chunk generati.
2026-09-11 20:28:30,109 - ingest - INFO - Indicizzazione (embeddings + BM25 + metadati + bounding box immagini)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 2129.41it/s]
[TIMING][--------] embedding.caricamento_modello: 5993 ms
2026-09-11 20:30:10,865 - ingest - INFO - Completato: printf_f v03 (modello PRINT! F)
```

`run --rm` avvia un container temporaneo che esegue il comando e poi viene
eliminato. Al primo avvio scarica i modelli nel volume `hf-cache`.
L'indice viene scritto in `data/vectorstore/` sull'host e resta disponibile dopo
la chiusura del container. Un indice costruito fuori da Docker viene riusato
così com'è: in quel caso questo passo si può saltare.

### Avvio dell'applicazione

```bash
docker compose up
```

L'app è raggiungibile su http://localhost:8501. All'avvio la pagina mostra
"Preparazione dei modelli...", il tempo di caricarli su CPU.
Per fermarla: `Ctrl+C`, oppure da
un'altra shell:

```bash
docker compose down
```

Se la porta 8501 è già occupata, per esempio dall'app avviata fuori da Docker,
Compose si ferma con `address already in use`. In quel caso si chiude l'altra
app, oppure si cambia la porta dell'host in `compose.yaml` (`"8502:8501"`).

Dopo una modifica al codice, ricostruzione e avvio si fanno insieme:

```bash
docker compose up --build
```

### Altri comandi

Qualsiasi script del repository si esegue allo stesso modo, per esempio la
valutazione end-to-end (richiede `ANTHROPIC_API_KEY`):

```bash
docker compose run --rm app python scripts/eval.py
```

### Pulizia

Rimuove container, immagine e volume dei modelli. La cartella `data/` resta
intatta sull'host:

```bash
docker compose down --rmi local --volumes
```
