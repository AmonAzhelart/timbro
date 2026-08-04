"""Verifica del token Hugging Face e dell'accesso ai repository gated.

I modelli pyannote richiedono, oltre al token, l'accettazione esplicita delle
condizioni d'uso sul sito. Senza quella la diarizzazione non fallisce: gira e
restituisce etichette vuote. Meglio dirlo all'utente prima, non dopo.
"""

from __future__ import annotations

import logging

import httpx

from .config import settings

log = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api"

#: Repo che devono essere accessibili perché la diarizzazione funzioni.
REQUIRED_REPOS = ("pyannote/segmentation-3.0",)


def verify(token: str | None = None) -> dict:
    """Restituisce un report leggibile dall'interfaccia."""
    token = (token or settings.hf_token or "").strip()
    if not token:
        return {
            "ok": False,
            "message": "Nessun token configurato.",
            "user": None,
            "repos": [],
        }

    headers = {"Authorization": f"Bearer {token}"}
    repos_to_check = list(dict.fromkeys((settings.diarization_model, *REQUIRED_REPOS)))

    try:
        with httpx.Client(timeout=20, headers=headers) as client:
            whoami = client.get(f"{HF_API}/whoami-v2")
            if whoami.status_code in (401, 403):
                return {
                    "ok": False,
                    "message": "Token non valido o revocato.",
                    "user": None,
                    "repos": [],
                }
            whoami.raise_for_status()
            user = whoami.json().get("name")

            repos = []
            for repo in repos_to_check:
                repos.append(_check_repo(client, repo))

    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "message": f"Hugging Face non raggiungibile: {exc}",
            "user": None,
            "repos": [],
        }

    blocked = [r["repo"] for r in repos if not r["accessible"]]
    if blocked:
        return {
            "ok": False,
            "user": user,
            "repos": repos,
            "message": (
                "Token valido, ma devi accettare le condizioni d'uso di: "
                + ", ".join(blocked)
                + ". Apri la pagina del modello su Hugging Face e clicca "
                "\"Agree and access repository\"."
            ),
        }

    return {
        "ok": True,
        "user": user,
        "repos": repos,
        "message": f"Token valido per l'utente {user}. Accesso ai modelli confermato.",
    }


def _check_repo(client: httpx.Client, repo: str) -> dict:
    try:
        response = client.get(f"{HF_API}/models/{repo}")
    except httpx.HTTPError as exc:
        return {"repo": repo, "accessible": False, "detail": str(exc)}

    if response.status_code == 200:
        return {"repo": repo, "accessible": True, "detail": "accesso confermato"}
    if response.status_code in (401, 403):
        return {"repo": repo, "accessible": False, "detail": "condizioni d'uso non accettate"}
    if response.status_code == 404:
        return {"repo": repo, "accessible": False, "detail": "repository inesistente"}
    return {"repo": repo, "accessible": False, "detail": f"HTTP {response.status_code}"}
