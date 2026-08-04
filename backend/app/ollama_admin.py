"""Gestione dei modelli Ollama dall'interfaccia: elenco, download, rimozione."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import httpx

from .config import settings

log = logging.getLogger(__name__)

#: Selezione curata proposta nella UI. `vram_gb` è indicativa a quantizzazione Q4.
SUGGESTED_LLMS: list[dict[str, Any]] = [
    {
        "name": "qwen3:8b",
        "label": "Qwen 3 · 8B",
        "vram_gb": 6,
        "note": "Default consigliato: ottimo italiano, leggero.",
    },
    {
        "name": "qwen3:14b",
        "label": "Qwen 3 · 14B",
        "vram_gb": 10,
        "note": "Action point e responsabili più accurati.",
    },
    {
        "name": "gemma3:12b",
        "label": "Gemma 3 · 12B",
        "vram_gb": 9,
        "note": "Molto solido sul multilingua.",
    },
    {
        "name": "gemma3:4b",
        "label": "Gemma 3 · 4B",
        "vram_gb": 4,
        "note": "Per GPU piccole o riunioni brevi.",
    },
    {
        "name": "llama3.1:8b",
        "label": "Llama 3.1 · 8B",
        "vram_gb": 5,
        "note": "Alternativa collaudata.",
    },
    {
        "name": "mistral-small:24b",
        "label": "Mistral Small · 24B",
        "vram_gb": 15,
        "note": "Sintesi molto curate, richiede GPU capiente.",
    },
    {
        "name": "gemma3:27b",
        "label": "Gemma 3 · 27B",
        "vram_gb": 17,
        "note": "Da 24 GB in su. Ottimo italiano, verbali molto solidi.",
    },
    {
        "name": "qwen3:32b",
        "label": "Qwen 3 · 32B",
        "vram_gb": 19,
        "note": "Il massimo che entra in 24 GB. Metti OLLAMA_KEEP_ALIVE=0.",
    },
]


def _url(path: str) -> str:
    return f"{settings.ollama_base_url.rstrip('/')}{path}"


# ---------------------------------------------------------------------------
# Lettura
# ---------------------------------------------------------------------------
def list_models() -> list[dict[str, Any]]:
    """Modelli già scaricati, con dimensione su disco."""
    with httpx.Client(timeout=15) as client:
        data = client.get(_url("/api/tags")).json()
    out = []
    for m in data.get("models", []):
        out.append(
            {
                "name": m.get("name", ""),
                "size_gb": round((m.get("size") or 0) / 1e9, 1),
                "family": (m.get("details") or {}).get("family", ""),
            }
        )
    return sorted(out, key=lambda m: m["name"])


def installed_names() -> list[str]:
    try:
        return [m["name"] for m in list_models()]
    except Exception as exc:
        log.warning("Elenco modelli Ollama non disponibile: %s", exc)
        return []


def is_installed(model: str) -> bool:
    names = installed_names()
    if model in names:
        return True
    # "qwen3:8b" copre anche il tag implicito "qwen3:latest"
    base = model.split(":")[0]
    return any(n.split(":")[0] == base for n in names) if ":" not in model else False


# ---------------------------------------------------------------------------
# Download con avanzamento
# ---------------------------------------------------------------------------
_pull_lock = threading.Lock()
_pull_state: dict[str, Any] = {
    "active": False,
    "model": None,
    "status": "",
    "percent": 0,
    "error": None,
    "done": False,
}


def pull_status() -> dict[str, Any]:
    with _pull_lock:
        return dict(_pull_state)


def start_pull(model: str) -> dict[str, Any]:
    """Avvia il download in background. Un solo download alla volta."""
    with _pull_lock:
        if _pull_state["active"]:
            raise RuntimeError(f"Download già in corso: {_pull_state['model']}")
        _pull_state.update(
            active=True, model=model, status="avvio", percent=0, error=None, done=False
        )

    threading.Thread(target=_pull_worker, args=(model,), daemon=True).start()
    return pull_status()


def _pull_worker(model: str) -> None:
    try:
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST", _url("/api/pull"), json={"model": model, "stream": True}
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if error := event.get("error"):
                        raise RuntimeError(error)

                    total = event.get("total") or 0
                    completed = event.get("completed") or 0
                    percent = int(completed * 100 / total) if total else 0
                    with _pull_lock:
                        _pull_state["status"] = event.get("status", "")
                        if total:
                            _pull_state["percent"] = min(percent, 99)

        with _pull_lock:
            _pull_state.update(active=False, done=True, percent=100, status="completato")
        log.info("Modello %s scaricato", model)

    except Exception as exc:
        log.error("Download di %s fallito: %s", model, exc)
        with _pull_lock:
            _pull_state.update(active=False, done=True, error=str(exc), status="errore")


def delete_model(model: str) -> None:
    if model == settings.ollama_model:
        raise ValueError("Non puoi rimuovere il modello attualmente in uso")
    with httpx.Client(timeout=60) as client:
        response = client.request("DELETE", _url("/api/delete"), json={"model": model})
        response.raise_for_status()
    log.info("Modello %s rimosso", model)
