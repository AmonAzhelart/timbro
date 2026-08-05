# Base con CUDA 12.4 + cuDNN 9 già pronti: evita di dover installare
# manualmente le librerie NVIDIA richieste da faster-whisper/ctranslate2.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

# ffmpeg: decodifica qualsiasi formato audio/video
# libsndfile1: richiesto da pyannote/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# NB: il contesto di build è la ROOT del progetto (vedi docker-compose.yml),
# così possiamo copiare sia backend/ sia frontend/.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/tools ./tools
COPY frontend ./frontend

# FullCalendar per la vista a calendario dei task. Viene scaricato qui e
# servito da noi: a runtime l'applicazione resta completamente locale, che è
# il punto di questo progetto — una CDN nel tag <script> significherebbe
# calendario rotto senza rete.
#
# Il download NON blocca il build: se fallisce, l'interfaccia se ne accorge e
# ripiega sulla vista a elenco, che resta comunque completa. Preferisco un
# calendario mancante a un'immagine che non si costruisce.
ARG FULLCALENDAR_VERSION=6.1.15
RUN mkdir -p frontend/vendor && \
    (curl -fsSL --retry 3 --max-time 60 \
        "https://cdn.jsdelivr.net/npm/fullcalendar@${FULLCALENDAR_VERSION}/index.global.min.js" \
        -o frontend/vendor/fullcalendar.js \
     || curl -fsSL --retry 3 --max-time 60 \
        "https://cdn.jsdelivr.net/npm/fullcalendar@${FULLCALENDAR_VERSION}/dist/index.global.min.js" \
        -o frontend/vendor/fullcalendar.js \
     || echo "FullCalendar non scaricato: la vista a calendario ripiegherà sull'elenco.") && \
    ls -l frontend/vendor/ || true

# Controlli statici su rotte, import e coerenza con la UI: se qualcosa è
# malformato la build fallisce subito, invece di lasciare il container in
# crash-loop dopo il deploy.
RUN python tools/preflight.py

RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
