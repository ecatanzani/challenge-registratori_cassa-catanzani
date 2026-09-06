# Installazione

Istruzioni per creare un ambiente pulito ed eseguire l'ingestion dei manuali.

## 1. Prerequisiti

**Python 3.11 o superiore.**

```bash
python3 --version
```

**Tesseract**, usato per l'OCR di fallback sulle pagine scansionate. Non è un
pacchetto Python: `pip` non lo installa e va messo a parte.

```bash
brew install tesseract tesseract-lang            # macOS
sudo apt install tesseract-ocr tesseract-ocr-ita # Debian/Ubuntu
```

Senza Tesseract il resto funziona comunque, ma ogni pagina scansionata viene
contata come `Fallito` nel log dell'ingestion e il suo testo non entra
nell'indice.

## 2. Creazione dell'ambiente virtuale

Dalla radice del repository:

```bash
python3 -m venv .venv
```

Attivazione (da ripetere a ogni nuova shell):

```bash
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
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

Sono circa 120 pacchetti fra dirette e transitive: `torch`,
`sentence-transformers` e `chromadb` costituiscono la quasi totalità del peso,
per qualche GB su disco. La prima installazione richiede alcuni minuti.

A installazione completata, configurazione ed esecuzione sono descritte nel
[README](README.md).
