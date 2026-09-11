FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ita \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py app.py ingest.py ./
COPY src ./src
COPY scripts ./scripts
COPY eval ./eval

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.fileWatcherType", "none", \
     "--server.address", "0.0.0.0", "--server.headless", "true"]
