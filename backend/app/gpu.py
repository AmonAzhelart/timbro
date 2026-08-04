"""Lettura dell'occupazione di memoria della GPU.

Il problema: Whisper e Ollama girano in **container distinti** sulla stessa
scheda. Interrogando CUDA dall'interno del container `api` si ottiene spesso
solo la memoria del proprio contesto — su Docker Desktop/WSL2 in particolare —
quindi il modello caricato da Ollama non compare e il totale resta inchiodato
al valore di riposo.

Per questo leggiamo da tre fonti indipendenti e componiamo:

* **NVML** (driver NVIDIA): occupazione dell'intera scheda. È la più
  attendibile quando disponibile.
* **torch**: memoria riservata da questo processo, cioè Whisper e pyannote.
* **Ollama `/api/ps`**: `size_vram` del modello caricato. Arriva via HTTP dal
  suo container, quindi è immune al problema di visibilità.

Il valore mostrato è il massimo fra quanto riporta il dispositivo e la somma
delle quote note: così, se il conteggio a livello di scheda sottostima, il
totale resta comunque veritiero.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

GB = 1024**3


# ---------------------------------------------------------------------------
# Fonti
# ---------------------------------------------------------------------------
def _from_nvml() -> dict[str, Any] | None:
    """Occupazione dell'intera scheda letta dal driver."""
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        log.debug("NVML non inizializzabile: %s", exc)
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        return {
            "source": "nvml",
            "name": name,
            "total": int(info.total),
            "used": int(info.used),
        }
    except Exception as exc:
        log.debug("Lettura NVML non riuscita: %s", exc)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _from_torch() -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        return {
            "source": "torch",
            "name": torch.cuda.get_device_properties(index).name,
            "total": int(total),
            "used": int(total - free),
            "process_reserved": int(torch.cuda.memory_reserved(index)),
            "process_allocated": int(torch.cuda.memory_allocated(index)),
        }
    except Exception as exc:
        log.debug("Lettura torch non riuscita: %s", exc)
        return None


def _from_ollama() -> dict[str, Any]:
    """Memoria occupata dai modelli che Ollama ha effettivamente in VRAM."""
    try:
        import httpx

        from .config import settings

        url = f"{settings.ollama_base_url.rstrip('/')}/api/ps"
        with httpx.Client(timeout=5) as client:
            data = client.get(url).json()
    except Exception as exc:
        log.debug("Ollama /api/ps non raggiungibile: %s", exc)
        return {"vram": 0, "models": [], "reachable": False}

    models = []
    total = 0
    for m in data.get("models", []):
        vram = int(m.get("size_vram") or 0)
        total += vram
        models.append({"name": m.get("name", ""), "vram_gb": round(vram / GB, 2)})
    return {"vram": total, "models": models, "reachable": True}


# ---------------------------------------------------------------------------
def stats() -> dict[str, Any]:
    device = _from_nvml()
    torch_info = _from_torch()
    if device is None:
        device = torch_info

    if device is None:
        return {
            "available": False,
            "reason": "Nessuna GPU CUDA visibile da questo container",
        }

    ollama = _from_ollama()
    transcription = int((torch_info or {}).get("process_reserved") or 0)
    llm = int(ollama["vram"])

    total = int(device["total"])
    # Se il dato di scheda sottostima (container che non vede l'altro processo),
    # la somma delle quote note è più vicina al vero.
    device_used = int(device["used"])
    used = max(device_used, transcription + llm)
    used = min(used, total)

    return {
        "available": True,
        "name": device.get("name", "GPU"),
        "source": device["source"],
        "total_gb": round(total / GB, 2),
        "used_gb": round(used / GB, 2),
        "free_gb": round(max(total - used, 0) / GB, 2),
        "percent": round(used * 100 / total) if total else 0,
        # Ripartizione: quanto sta usando la trascrizione e quanto l'LLM
        "transcription_gb": round(transcription / GB, 2),
        "llm_gb": round(llm / GB, 2),
        "other_gb": round(max(used - transcription - llm, 0) / GB, 2),
        "llm_models": ollama["models"],
        "ollama_reachable": ollama["reachable"],
        # Utile a capire se il dato di scheda è attendibile in questo setup
        "device_reported_gb": round(device_used / GB, 2),
        "undercounting": device_used < transcription + llm,
    }
