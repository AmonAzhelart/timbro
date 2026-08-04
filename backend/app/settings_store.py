"""Persistenza delle impostazioni modificate da interfaccia.

Precedenza dei valori (dal più debole al più forte):

    default nel codice  <  variabili d'ambiente (.env)  <  data/settings.json

`settings.json` contiene SOLO i campi realmente modificati dalla UI, così
chi preferisce lavorare da `.env` continua a farlo per tutto il resto.
Il file vive in `data/`, quindi sopravvive a rebuild e riavvii dei container.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from .config import settings
from .models import AppSettings

log = logging.getLogger(__name__)

_lock = threading.RLock()

#: Campi che l'interfaccia può modificare. Tutto il resto resta fuori portata.
EDITABLE: tuple[str, ...] = (
    "hf_token",
    "whisper_model",
    "whisper_compute_type",
    "whisper_batch_size",
    "whisper_language",
    "diarization_model",
    # Argomento, termini e numero di interlocutori sono proprietà della singola
    # riunione, non della configurazione: stanno sul job (vedi models.JobOptions).
    "beam_size",
    "no_speech_threshold",
    "vad_onset",
    "vad_offset",
    "multilingual",
    "audio_preprocess",
    "auto_context",
    "ollama_base_url",
    "ollama_model",
    "ollama_num_ctx",
    "ollama_timeout_s",
    "ollama_keep_alive",
    "chunk_chars",
    "max_upload_mb",
    "delete_audio_after",
)

#: Mai restituiti in chiaro dall'API.
SECRET_FIELDS: frozenset[str] = frozenset({"hf_token"})

#: Se cambia uno di questi, i modelli in VRAM vanno ricaricati.
MODEL_FIELDS: frozenset[str] = frozenset(
    {
        "whisper_model",
        "whisper_compute_type",
        "whisper_language",
        "diarization_model",
        "hf_token",
        # Le opzioni di decodifica sono congelate dentro la pipeline WhisperX
        # al momento del caricamento: senza ricarica non avrebbero effetto.
        "beam_size",
        "no_speech_threshold",
        "vad_onset",
        "vad_offset",
        "multilingual",
    }
)

VALID_COMPUTE_TYPES = {"float16", "float32", "int8_float16", "int8", "bfloat16"}


# ---------------------------------------------------------------------------
# I/O su file
# ---------------------------------------------------------------------------
def _path() -> Path:
    return settings.data_dir / "settings.json"


def _read() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in EDITABLE}
    except Exception as exc:
        log.error("settings.json illeggibile (%s): uso i valori d'ambiente", exc)
        return {}


def _write(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    try:  # il file contiene il token: niente lettura per gli altri utenti
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - filesystem che non supporta chmod
        pass


# ---------------------------------------------------------------------------
# Applicazione a runtime
# ---------------------------------------------------------------------------
def _apply(data: dict[str, Any]) -> None:
    """Scrive i valori sul singleton `settings`, visibile a tutti i moduli."""
    for key, value in data.items():
        if key in EDITABLE:
            setattr(settings, key, value)

    # huggingface_hub legge il token anche dall'ambiente per i repo gated
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token


def load_and_apply() -> dict[str, Any]:
    """Chiamata all'avvio, prima che il worker parta."""
    with _lock:
        data = _read()
        _apply(data)
        if data:
            log.info("Impostazioni da interfaccia applicate: %s", ", ".join(sorted(data)))
        else:
            _apply({})  # propaga comunque il token che arriva da .env
        return data


# ---------------------------------------------------------------------------
# Lettura / scrittura dall'API
# ---------------------------------------------------------------------------
def _mask(token: str) -> str:
    if not token:
        return ""
    return f"{token[:3]}…{token[-4:]}" if len(token) > 10 else "•" * len(token)


def current() -> AppSettings:
    with _lock:
        overridden = sorted(_read().keys())
        return AppSettings(
            hf_token_set=bool(settings.hf_token),
            hf_token_preview=_mask(settings.hf_token),
            whisper_model=settings.whisper_model,
            whisper_compute_type=settings.whisper_compute_type,
            whisper_batch_size=settings.whisper_batch_size,
            whisper_language=settings.whisper_language,
            diarization_model=settings.diarization_model,
            beam_size=settings.beam_size,
            no_speech_threshold=settings.no_speech_threshold,
            vad_onset=settings.vad_onset,
            vad_offset=settings.vad_offset,
            multilingual=settings.multilingual,
            audio_preprocess=settings.audio_preprocess,
            auto_context=settings.auto_context,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            ollama_num_ctx=settings.ollama_num_ctx,
            ollama_timeout_s=settings.ollama_timeout_s,
            ollama_keep_alive=settings.ollama_keep_alive,
            chunk_chars=settings.chunk_chars,
            max_upload_mb=settings.max_upload_mb,
            delete_audio_after=settings.delete_audio_after,
            overridden=overridden,
        )


def update(patch: dict[str, Any]) -> tuple[AppSettings, bool]:
    """Applica e salva un aggiornamento parziale.

    Restituisce `(impostazioni, serve_ricarica_modelli)`.
    Le chiavi assenti da `patch` restano invariate; una chiave presente con
    valore `None` (o stringa vuota) azzera il campo tornando al default d'ambiente.
    """
    patch = {k: v for k, v in patch.items() if k in EDITABLE}
    _validate(patch)

    with _lock:
        stored = _read()
        defaults = _env_defaults()
        touched_models = False

        for key, value in patch.items():
            cleared = value is None or (isinstance(value, str) and not value.strip())
            new_value = defaults[key] if cleared else _coerce(key, value)

            if getattr(settings, key, None) != new_value and key in MODEL_FIELDS:
                touched_models = True

            # Salviamo solo ciò che si discosta da .env: il file resta minimale
            # e i campi lasciati com'erano continuano a seguire l'ambiente.
            if cleared or new_value == defaults[key]:
                stored.pop(key, None)
            else:
                stored[key] = new_value

            setattr(settings, key, new_value)

        _write(stored)
        _apply(stored)

        if touched_models:
            _invalidate_models()

        log.info("Impostazioni aggiornate: %s", ", ".join(sorted(patch)) or "nessun campo")
        return current(), touched_models


def _invalidate_models() -> None:
    """Scarica i modelli dalla VRAM: al job successivo verranno ricaricati."""
    try:
        from . import transcribe

        transcribe.release_gpu()
        log.info("Cache modelli invalidata dopo il cambio impostazioni")
    except Exception as exc:  # pragma: no cover
        log.warning("Impossibile invalidare la cache dei modelli: %s", exc)


def _env_defaults() -> dict[str, Any]:
    """Valori che i campi avrebbero senza override da interfaccia."""
    from .config import Settings

    fresh = Settings()  # rilegge ambiente e .env
    return {key: getattr(fresh, key) for key in EDITABLE}


INT_FIELDS = frozenset({
    "whisper_batch_size", "ollama_num_ctx", "ollama_timeout_s", "chunk_chars",
    "max_upload_mb", "beam_size",
})
FLOAT_FIELDS = frozenset({"no_speech_threshold", "vad_onset", "vad_offset"})
BOOL_FIELDS = frozenset({
    "delete_audio_after", "multilingual", "audio_preprocess", "auto_context",
})


def _coerce(key: str, value: Any) -> Any:
    """Normalizza i tipi in arrivo dal JSON."""
    if key in INT_FIELDS:
        return int(value)
    if key in FLOAT_FIELDS:
        return float(value)
    if key in BOOL_FIELDS:
        return bool(value)
    if isinstance(value, str):
        return value.strip()
    return value


def _validate(patch: dict[str, Any]) -> None:
    """Controlli che Pydantic da solo non può fare (coerenza fra campi)."""
    ct = patch.get("whisper_compute_type")
    if ct and ct not in VALID_COMPUTE_TYPES:
        raise ValueError(
            f"compute_type '{ct}' non valido. Ammessi: {', '.join(sorted(VALID_COMPUTE_TYPES))}"
        )

    url = patch.get("ollama_base_url")
    if url and not str(url).startswith(("http://", "https://")):
        raise ValueError("L'indirizzo di Ollama deve iniziare con http:// o https://")

    keep = patch.get("ollama_keep_alive")
    if keep is not None and str(keep).strip():
        # Ollama accetta un numero di secondi oppure una durata tipo "5m"/"1h".
        if not re.fullmatch(r"-?\d+[smh]?", str(keep).strip()):
            raise ValueError(
                "Permanenza in VRAM non valida: usa un numero di secondi "
                "(0, 300) oppure una durata (30s, 5m, 1h). -1 = sempre in memoria."
            )

    token = patch.get("hf_token")
    if token and not str(token).startswith("hf_"):
        raise ValueError("Il token Hugging Face deve iniziare con 'hf_'")
