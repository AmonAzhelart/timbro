# Ambiente ISOLATO per la separazione delle voci.
#
# Il pipeline pyannote/speech-separation-ami-1.0 è nella "legacy collection"
# e vuole pyannote.audio 3.3.x; WhisperX 3.8.6 vuole la 4.x. Le due versioni
# non convivono, quindi questo servizio ha il proprio ambiente pinnato e
# comunica con l'api via HTTP.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# `pesq` arriva come dipendenza transitiva di asteroid e non ha wheel
# precompilate: va compilata da sorgente, e l'immagine base non ha un
# compilatore. Installiamo il toolchain, compiliamo, e lo rimuoviamo nello
# stesso layer per non trascinarci ~250 MB nell'immagine finale.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app.py .

RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]
