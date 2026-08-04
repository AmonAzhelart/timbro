"""Client verso il servizio di separazione delle voci.

Il servizio gira in un container a parte (pyannote 3.3.x, incompatibile con la
4.x di WhisperX). Qui ci limitiamo a chiamarlo e a gestire il caso in cui non
ci sia: la separazione è opzionale e la sua assenza non deve mai far fallire
un job, solo produrre un avviso.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import settings

log = logging.getLogger(__name__)


def _url(path: str) -> str:
    return f"{settings.separator_url.rstrip('/')}{path}"


def available() -> dict[str, Any]:
    try:
        with httpx.Client(timeout=5) as client:
            data = client.get(_url("/health")).json()
        return {"reachable": True, **data}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def diagnose() -> dict[str, Any]:
    """Chiede al servizio di caricare davvero il modello e riportare l'esito.

    Il token viaggia nella richiesta: se è stato impostato dall'interfaccia
    vive in `data/settings.json`, che il container del separatore non legge.
    """
    try:
        with httpx.Client(timeout=300) as client:
            response = client.post(_url("/diagnose"), json={"hf_token": settings.hf_token})
            return {"reachable": True, **response.json()}
    except Exception as exc:
        return {
            "reachable": False,
            "ok": False,
            "error": (
                f"Servizio di separazione non raggiungibile ({exc}). "
                "Avvialo con: docker compose --profile separation up -d"
            ),
        }


def separate(audio_path: str, job_id: str) -> dict[str, Any]:
    """Restituisce `{ok, sources: [{speaker, path, speech_s}], diarization, error}`."""
    try:
        with httpx.Client(timeout=settings.separator_timeout_s) as client:
            response = client.post(
                _url("/separate"),
                json={
                    "audio_path": audio_path,
                    "job_id": job_id,
                    "release_after": True,
                    "hf_token": settings.hf_token,
                },
            )
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        log.error("Separazione non riuscita: %s", exc)
        return {
            "ok": False,
            "sources": [],
            "error": (
                f"Servizio di separazione non disponibile ({exc}). "
                "Avvialo con: docker compose --profile separation up -d"
            ),
        }
