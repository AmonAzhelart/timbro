"""Servizio di separazione delle voci sovrapposte.

Perché è un container a parte: il pipeline `pyannote/speech-separation-ami-1.0`
richiede **pyannote.audio 3.3.x**, mentre WhisperX 3.8.6 richiede la **4.x**.
Nello stesso ambiente non convivono. Isolarlo qui permette di pinnare la
versione giusta senza toccare la pipeline principale, che continua a
funzionare identica quando la separazione è spenta o questo servizio è fermo.

Contratto: riceve il percorso di un audio dentro `/data` (volume condiviso con
l'api), restituisce i percorsi delle tracce separate, una per voce, più gli
intervalli di diarizzazione.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("separator")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
OUT_DIR = DATA_DIR / "separated"
MODEL = os.environ.get("SEPARATION_MODEL", "pyannote/speech-separation-ami-1.0")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

app = FastAPI(title="Timbro separator", version="1.0.0")

_lock = threading.Lock()
_pipeline: Any = None


# ---------------------------------------------------------------------------
def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load(hf_token: str = ""):
    """Carica il pipeline una volta sola, tenendolo in memoria.

    Il token arriva preferibilmente dalla richiesta: l'api è l'unica a
    conoscere quello effettivo, perché può essere stato impostato
    dall'interfaccia e salvato in `data/settings.json`, che questo
    container non legge. `HF_TOKEN` d'ambiente resta come ripiego.
    """
    global _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline

        from pyannote.audio import Pipeline

        token = (hf_token or HF_TOKEN or "").strip() or None
        if not token:
            raise RuntimeError(
                "Nessun token Hugging Face disponibile. Impostalo in "
                "⚙ Impostazioni oppure valorizza HF_TOKEN nel file .env."
            )
        log.info("Carico %s (token %s…)", MODEL, token[:6])

        # pyannote 3.3 usa `use_auth_token`; teniamo il fallback su `token`
        # nel caso di future revisioni della firma.
        try:
            pipeline = Pipeline.from_pretrained(MODEL, use_auth_token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(MODEL, token=token)

        if pipeline is None:
            raise RuntimeError(
                f"Pipeline '{MODEL}' non caricato: di norma significa che non hai "
                "accettato le condizioni d'uso su Hugging Face, oppure che il token "
                "non ha accesso. Servono le condizioni di speech-separation-ami-1.0 "
                "E di separation-ami-1.0."
            )

        device = _device()
        if device == "cuda":
            import torch

            pipeline.to(torch.device("cuda"))
        log.info("Pipeline di separazione pronto su %s", device)
        _pipeline = pipeline
        return _pipeline


def release() -> None:
    """Libera la VRAM: la GPU è condivisa con Whisper e Ollama."""
    global _pipeline
    with _lock:
        _pipeline = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
def _decode_to_wav(source: Path) -> Path:
    """Porta qualunque formato a WAV PCM 16 kHz mono, usando ffmpeg.

    libsndfile — su cui si appoggia pyannote — legge solo formati non
    compressi o poco altro: m4a, mp3, aac e i contenitori video lo fanno
    fallire con "Format not recognised". ffmpeg li apre tutti.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="separator-"))
    out = tmp_dir / "input.wav"
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-i", str(source),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(
            f"Impossibile decodificare l'audio: {detail[-1] if detail else exc}"
        ) from exc
    log.info("Audio decodificato in WAV 16 kHz mono: %s", source.name)
    return out


class TokenRequest(BaseModel):
    #: Token effettivo, inviato dall'api: può venire da settings.json, che
    #: questo container non vede.
    hf_token: str = ""


class SeparateRequest(TokenRequest):
    audio_path: str = Field(description="Percorso dentro /data, condiviso con l'api")
    job_id: str = ""
    #: Scarica il modello dalla VRAM al termine (default: sì, la GPU è condivisa)
    release_after: bool = True


class Source(BaseModel):
    speaker: str
    path: str
    speech_s: float


class SeparateResponse(BaseModel):
    ok: bool
    sources: list[Source] = Field(default_factory=list)
    diarization: list[dict] = Field(default_factory=list)
    error: str | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL,
        "device": _device(),
        "token_set": bool(HF_TOKEN),
        "loaded": _pipeline is not None,
    }


@app.post("/diagnose")
def diagnose(payload: TokenRequest | None = None) -> dict:
    """Carica davvero il pipeline e riporta l'errore esatto, senza indovinare."""
    token = ((payload.hf_token if payload else "") or HF_TOKEN or "").strip()
    info: dict[str, Any] = {
        "model": MODEL,
        "device": _device(),
        "token_set": bool(token),
        "token_source": "richiesta" if (payload and payload.hf_token) else
                        ("ambiente" if HF_TOKEN else "assente"),
    }
    try:
        import pyannote.audio

        info["pyannote_version"] = pyannote.audio.__version__
    except Exception as exc:
        info.update(ok=False, error=f"pyannote.audio non importabile: {exc}")
        return info

    if not token:
        info.update(
            ok=False,
            error="Nessun token Hugging Face: impostalo in ⚙ Impostazioni "
                  "oppure in HF_TOKEN dentro .env.",
        )
        return info

    try:
        _load(token)
    except Exception as exc:
        info.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        return info

    info.update(ok=True, error=None)
    return info


@app.post("/separate", response_model=SeparateResponse)
def separate(req: SeparateRequest) -> SeparateResponse:
    source = Path(req.audio_path)
    if not source.exists():
        raise HTTPException(404, f"Audio non trovato: {source}")

    try:
        pipeline = _load(req.hf_token)
    except Exception as exc:
        log.exception("Caricamento del pipeline fallito")
        return SeparateResponse(ok=False, error=f"{type(exc).__name__}: {exc}")

    decoded: Path | None = None
    try:
        import numpy as np
        import scipy.io.wavfile

        # pyannote legge l'audio con libsndfile, che non conosce AAC/m4a/mp3:
        # su quei formati fallisce con "Format not recognised". WhisperX non ha
        # il problema perché decodifica con ffmpeg. Lo facciamo anche qui,
        # ottenendo per giunta i 16 kHz mono che il modello si aspetta.
        decoded = _decode_to_wav(source)
        log.info("Separazione di %s", source.name)
        diarization, sources = pipeline(str(decoded))

        job_dir = OUT_DIR / (req.job_id or uuid.uuid4().hex[:12])
        job_dir.mkdir(parents=True, exist_ok=True)

        labels = list(diarization.labels())
        out: list[Source] = []
        for index, speaker in enumerate(labels):
            if index >= sources.data.shape[1]:
                break
            track = np.asarray(sources.data[:, index], dtype=np.float32)
            peak = float(np.max(np.abs(track))) if track.size else 0.0
            if peak > 0:
                # Il pipeline può produrre picchi oltre 1.0: normalizziamo per
                # evitare la distorsione che manderebbe in crisi Whisper.
                track = track / max(peak, 1.0)
            path = job_dir / f"{speaker}.wav"
            scipy.io.wavfile.write(str(path), 16000, (track * 32767).astype(np.int16))

            speech_s = sum(
                seg.duration for seg, _, spk in diarization.itertracks(yield_label=True)
                if spk == speaker
            )
            out.append(Source(speaker=speaker, path=str(path), speech_s=round(speech_s, 2)))

        turns = [
            {"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": spk}
            for seg, _, spk in diarization.itertracks(yield_label=True)
        ]
        log.info("Separazione completata: %s tracce, %s turni", len(out), len(turns))
        return SeparateResponse(ok=True, sources=out, diarization=turns)

    except Exception as exc:
        log.exception("Separazione fallita")
        return SeparateResponse(ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if decoded is not None and decoded != source:
            shutil.rmtree(decoded.parent, ignore_errors=True)
        if req.release_after:
            release()
