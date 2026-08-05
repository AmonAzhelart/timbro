"""Pipeline WhisperX: trascrizione -> allineamento word-level -> diarizzazione.

Punti chiave:
* i modelli ASR/allineamento restano in memoria fra un job e l'altro
  (caricarli costa ~20-40 s, non ha senso ripeterlo);
* la VRAM viene liberata esplicitamente prima di passare la palla all'LLM,
  altrimenti Whisper e Ollama si contendono la stessa GPU;
* la diarizzazione non fallisce mai in silenzio: un errore o zero segmenti
  producono un avviso esplicito che risale fino all'interfaccia.
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from . import overlap
from .config import settings

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, str], None]

# --- Import difensivi -------------------------------------------------------
import whisperx  # noqa: E402

try:  # WhisperX >= 3.8
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
except ImportError:  # pragma: no cover - WhisperX <= 3.7
    DiarizationPipeline = whisperx.DiarizationPipeline  # type: ignore[attr-defined]
    assign_word_speakers = whisperx.assign_word_speakers  # type: ignore[attr-defined]


class DiarizationError(RuntimeError):
    """La diarizzazione non ha prodotto etichette utilizzabili."""


class _ModelCache:
    """Cache thread-safe dei modelli pesanti."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._asr: Any = None
        self._asr_key: tuple | None = None
        self._align: dict[str, tuple[Any, Any]] = {}
        self._diarizer: Any = None
        self._diarizer_key: tuple | None = None

    # -- ASR ----------------------------------------------------------------
    def asr(self, device: str, *, model_name: str | None = None, options: dict | None = None):
        asr_options = options if options is not None else settings.asr_options()
        vad_options = settings.vad_options
        model_name = model_name or settings.whisper_model
        # Le opzioni fanno parte della chiave: cambiare vocabolario o beam size
        # deve ricostruire la pipeline, altrimenti resterebbero senza effetto.
        key = (
            model_name,
            device,
            settings.whisper_compute_type,
            settings.language,
            tuple(sorted(asr_options.items())),
            tuple(sorted(vad_options.items())),
        )
        with self._lock:
            if self._asr is None or self._asr_key != key:
                # Rilasciamo il modello precedente PRIMA di caricare il nuovo:
                # tenerli entrambi anche per un istante raddoppia il picco di
                # VRAM, ed è proprio quello che succede fra ricognizione e
                # trascrizione vera.
                if self._asr is not None:
                    self._asr = None
                    self._asr_key = None
                    _empty_cuda_cache()
                log.info(
                    "Carico modello Whisper %s (%s) | beam=%s prompt=%s hotwords=%s",
                    model_name,
                    device,
                    asr_options.get("beam_size"),
                    "sì" if asr_options.get("initial_prompt") else "no",
                    "sì" if asr_options.get("hotwords") else "no",
                )
                self._asr = whisperx.load_model(
                    model_name,
                    device=device,
                    compute_type=settings.whisper_compute_type,
                    language=settings.language,
                    asr_options=asr_options,
                    vad_options=vad_options,
                )
                self._asr_key = key
            return self._asr

    # -- Allineamento -------------------------------------------------------
    def align(self, language_code: str, device: str):
        with self._lock:
            if language_code not in self._align:
                log.info("Carico modello di allineamento per '%s'", language_code)
                self._align[language_code] = whisperx.load_align_model(
                    language_code=language_code, device=device
                )
            return self._align[language_code]

    # -- Diarizzazione ------------------------------------------------------
    def diarizer(self, device: str):
        key = (settings.diarization_model, device, settings.hf_token)
        with self._lock:
            if self._diarizer is None or self._diarizer_key != key:
                log.info("Carico pipeline di diarizzazione %s", settings.diarization_model)
                self._diarizer = build_diarizer(device)
                self._diarizer_key = key
            return self._diarizer

    def release_gpu(self) -> None:
        """Libera la VRAM tenendo i modelli fuori dalla GPU."""
        with self._lock:
            self._asr = None
            self._asr_key = None
            self._align.clear()
            self._diarizer = None
            self._diarizer_key = None
        _empty_cuda_cache()


def build_diarizer(device: str):
    """Istanzia `DiarizationPipeline` adattandosi alla firma della versione installata.

    WhisperX 3.8 ha rinominato il parametro del token da `use_auth_token` a
    `token` (allineandosi a pyannote 4). Passare quello sbagliato solleva
    `TypeError`, che è facile scambiare per "diarizzazione non disponibile":
    leggiamo la firma reale invece di tirare a indovinare, così eventuali
    errori che emergono sono quelli veri (token invalido, licenza non accettata).
    """
    params = inspect.signature(DiarizationPipeline.__init__).parameters
    kwargs: dict[str, Any] = {"device": device}
    token = settings.hf_token or None

    if "token" in params:
        kwargs["token"] = token
    elif "use_auth_token" in params:
        kwargs["use_auth_token"] = token
    else:  # pragma: no cover - firma inattesa
        raise DiarizationError(
            "La versione di WhisperX installata non accetta un token per la "
            "diarizzazione: firma sconosciuta di DiarizationPipeline."
        )

    if "model_name" in params and settings.diarization_model:
        kwargs["model_name"] = settings.diarization_model

    return DiarizationPipeline(**kwargs)


_cache = _ModelCache()


def _empty_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover
        pass


def pick_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def release_gpu() -> None:
    """Chiamata dalla pipeline prima di interrogare l'LLM."""
    _cache.release_gpu()


def diagnose() -> dict[str, Any]:
    """Prova a costruire la pipeline di diarizzazione e riporta l'esito esatto.

    Serve all'interfaccia: invece di lasciare l'utente davanti a una trascrizione
    senza speaker, gli diciamo qual è l'errore preciso.
    """
    device = pick_device()
    signature = str(inspect.signature(DiarizationPipeline.__init__))
    info: dict[str, Any] = {
        "device": device,
        "model": settings.diarization_model,
        "token_set": bool(settings.hf_token),
        "signature": signature,
    }
    if not settings.hf_token:
        info.update(ok=False, error="Nessun token Hugging Face configurato.")
        return info
    try:
        build_diarizer(device)
    except Exception as exc:
        info.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        return info
    info.update(ok=True, error=None)
    return info


# ---------------------------------------------------------------------------
# Diarizzazione
# ---------------------------------------------------------------------------
def _phase_progress(progress: ProgressFn, status: str, start: int, end: int, label: str):
    """Adatta un callback 0-100 di libreria alla nostra scala complessiva.

    WhisperX e pyannote riportano l'avanzamento della singola fase; qui lo
    rimappiamo sull'intervallo che quella fase occupa nel job.
    """
    def report(pct: float) -> None:
        value = start + int(max(0.0, min(pct, 100.0)) / 100 * (end - start))
        progress(status, value, f"{label} {int(pct)}%")

    return report


def diarize(
    audio: Any,
    result: dict[str, Any],
    device: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress: ProgressFn | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Applica la diarizzazione a un risultato già trascritto.

    Il numero atteso di interlocutori arriva dal job: è una proprietà della
    singola riunione, non una configurazione dell'applicazione.

    Restituisce `(result, report)`; `report` descrive sempre l'esito, anche
    quando va male, così il chiamante può mostrarlo invece di ignorarlo.
    """
    report: dict[str, Any] = {"ok": False, "error": None, "speakers": 0}

    try:
        diarizer = _cache.diarizer(device)
    except Exception as exc:
        report["error"] = _explain(exc)
        log.error("Diarizzazione non inizializzata: %s", exc)
        return result, report

    kwargs: dict[str, Any] = {}
    if min_speakers:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers:
        kwargs["max_speakers"] = int(max_speakers)

    if progress is not None:
        kwargs["progress_callback"] = _phase_progress(
            progress, "diarizing", 65, 72, "Riconoscimento voci"
        )

    try:
        diarize_segments = diarizer(audio, **kwargs)
    except TypeError:
        # Versioni senza `progress_callback`: riproviamo senza.
        kwargs.pop("progress_callback", None)
        diarize_segments = diarizer(audio, **kwargs)
    except Exception as exc:
        report["error"] = _explain(exc)
        log.error("Diarizzazione fallita: %s", exc)
        return result, report

    if diarize_segments is None or len(diarize_segments) == 0:
        report["error"] = (
            "La diarizzazione è andata a buon fine ma non ha prodotto alcun segmento. "
            "Di norma significa che il token Hugging Face non ha accesso al modello: "
            "verifica di avere accettato le condizioni d'uso con lo stesso account "
            "che ha generato il token."
        )
        log.error("Diarizzazione senza segmenti: probabile licenza non accettata")
        return result, report

    # `fill_nearest` etichetta anche le parole per cui la diarizzazione non ha
    # evidenza: copre tutto ma inventa, ed è la causa principale dei falsi
    # cambi di parlante. Predefinito: spento.
    result = assign_word_speakers(
        diarize_segments, result, fill_nearest=settings.diarization_fill_nearest
    )

    # Gli scambi rapidi finirebbero attribuiti a una sola persona: le etichette
    # per-parola sono più precise di quella per-segmento. La soglia evita che
    # una singola parola mal etichettata generi un turno inesistente.
    result["segments"] = overlap.split_on_speaker_change(
        result.get("segments", []),
        min_words=settings.speaker_change_min_words,
        min_seconds=settings.speaker_change_min_s,
    )

    speakers = {
        seg.get("speaker")
        for seg in result.get("segments", [])
        if seg.get("speaker")
    }
    report["speakers"] = len(speakers)
    if not speakers:
        report["error"] = (
            "Nessun segmento della trascrizione ha ricevuto un'etichetta voce, "
            "pur avendo ottenuto dati di diarizzazione."
        )
        return result, report

    report["ok"] = True
    report["overlaps"] = overlap.find_overlaps(diarize_segments)
    log.info("Diarizzazione completata: %s voci distinte", len(speakers))
    return result, report


def _explain(exc: Exception) -> str:
    """Traduce le eccezioni più comuni in un messaggio azionabile."""
    text = str(exc)
    low = text.lower()

    if "401" in text or "unauthorized" in low or "invalid" in low and "token" in low:
        return (
            "Token Hugging Face non valido o scaduto. Rigeneralo da "
            "huggingface.co/settings/tokens e reinseriscilo in ⚙ Impostazioni."
        )
    if "403" in text or "gated" in low or "awaiting a review" in low or "accept" in low:
        return (
            f"Accesso negato al modello '{settings.diarization_model}'. Apri la sua "
            "pagina su Hugging Face e accetta le condizioni d'uso con lo stesso "
            "account che ha generato il token."
        )
    if "404" in text or "not found" in low:
        return (
            f"Modello '{settings.diarization_model}' inesistente su Hugging Face. "
            "Controlla il nome in ⚙ Impostazioni."
        )
    if isinstance(exc, TypeError):
        return (
            f"Incompatibilità con la versione di WhisperX installata: {text}. "
            "Probabile disallineamento fra le dipendenze del container."
        )
    if "out of memory" in low:
        return (
            "VRAM insufficiente per la diarizzazione. Riduci il modello Whisper "
            "o il batch size in ⚙ Impostazioni."
        )
    return f"{type(exc).__name__}: {text}"


# ---------------------------------------------------------------------------
# Trascrizione completa
# ---------------------------------------------------------------------------
def probe_recorded_at(source: Path) -> str:
    """Quando la riunione è stata REGISTRATA, non quando è stata caricata.

    Senza questo dato «entro domattina» non è risolvibile: se carichi lunedì
    una registrazione di venerdì, «domani» cade sul giorno sbagliato.

    Tre fonti in ordine di attendibilità: i metadati del contenitore (i
    telefoni scrivono `creation_time` quasi sempre), la data di modifica del
    file, e in ultimo nulla — il chiamante ripiegherà sul caricamento.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries", "format_tags=creation_time:format=duration",
             str(source)],
            capture_output=True, text=True, timeout=30,
        )
        tags = (json.loads(out.stdout or "{}").get("format") or {}).get("tags") or {}
        raw = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
        if raw:
            # ffprobe restituisce UTC con la Z finale; datetime la accetta
            # solo dalla 3.11, e comunque qui serve l'ora locale della riunione.
            cleaned = raw.replace("Z", "+00:00")
            moment = datetime.fromisoformat(cleaned)
            if moment.tzinfo is not None:
                moment = moment.astimezone().replace(tzinfo=None)
            log.info("Data di registrazione dai metadati: %s", moment.isoformat())
            return moment.isoformat(timespec="seconds")
    except Exception as exc:
        log.debug("Metadati temporali non leggibili da %s: %s", source.name, exc)

    try:
        moment = datetime.fromtimestamp(source.stat().st_mtime)
        log.info("Data di registrazione dalla data del file: %s", moment.isoformat())
        return moment.isoformat(timespec="seconds")
    except Exception:
        return ""


def preprocess_audio(source: Path) -> tuple[Path, str | None]:
    """Passa-alto, riduzione rumore e normalizzazione del volume via ffmpeg.

    Su registrazioni ambientali (più persone, microfono lontano, ronzio di
    fondo) recupera parecchie parole. Su audio già pulito è ininfluente,
    quindi resta opzionale. Restituisce `(percorso, errore)`.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="timbro-"))
    out = tmp_dir / "preprocessed.wav"
    filters = (
        "highpass=f=80,"          # taglia ronzii e rumore di fondo sotto la voce
        "afftdn=nf=-25,"          # riduzione rumore nel dominio della frequenza
        "loudnorm=I=-16:TP=-1.5:LRA=11"  # livella i volumi fra chi è vicino e chi è lontano
    )
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-i", str(source),
        "-af", filters, "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    except FileNotFoundError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return source, "ffmpeg non disponibile: uso l'audio originale."
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        return source, f"Pre-elaborazione audio fallita: {detail[-1] if detail else exc}"
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return source, "Pre-elaborazione audio troppo lenta: uso l'audio originale."

    log.info("Audio pre-elaborato in %s", out)
    return out, None


def transcribe(
    audio_path: Path,
    progress: ProgressFn,
    options: Any = None,
    skip_recon: bool = False,
) -> dict[str, Any]:
    """Restituisce segmenti, lingua, durata, report di ricognizione e diarizzazione.

    `options` è il `JobOptions` della riunione: argomento, termini e numero
    atteso di interlocutori. `skip_recon` disattiva la ricognizione quando
    l'utente ha già scritto il contesto a mano.
    """
    from .models import JobOptions

    options = (options or JobOptions()).normalized()
    device = pick_device()
    compute_type = settings.whisper_compute_type
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"  # float16 su CPU non è supportato da ctranslate2
        settings.whisper_compute_type = compute_type
        log.warning("Nessuna GPU rilevata: passo a compute_type=int8 su CPU")

    notices: list[str] = []
    source = audio_path
    temp_dir: Path | None = None
    if settings.audio_preprocess:
        progress("transcribing", 5, "Pulizia e normalizzazione dell'audio")
        source, err = preprocess_audio(audio_path)
        if err:
            notices.append(err)
        elif source != audio_path:
            temp_dir = source.parent

    try:
        return _run(source, device, progress, notices, options, skip_recon)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def reconnoitre(audio: Any, duration_s: float, device: str) -> dict[str, Any]:
    """Fase 0: trascrizione veloce di alcuni estratti per dedurre il contesto.

    Usa un modello leggero su pochi minuti campionati, poi passa il testo grezzo
    all'LLM. Il risultato non finisce nella trascrizione finale: serve solo a
    costruire `initial_prompt` e `hotwords` per il passaggio vero.
    """
    from . import recon, summarize  # import locale: evita cicli a caricamento

    sampled = recon.sample_windows(audio, duration_s)
    sampled_s = len(sampled) / 16000.0

    # Modello leggero, decodifica rapida: qui la precisione non serve.
    draft_options = {"beam_size": 1, "best_of": 1, "no_speech_threshold": 0.6}
    model = _cache.asr(device, model_name=recon.RECON_MODEL, options=draft_options)
    draft_result = model.transcribe(
        sampled,
        batch_size=settings.whisper_batch_size,
        language=settings.language,
    )
    draft = recon.clean_draft(draft_result.get("segments", []))
    if not draft:
        return {"ok": False, "reason": "La ricognizione non ha prodotto testo utilizzabile."}

    inferred = summarize.infer_context(draft)
    terms = recon.filter_terms(inferred.get("terms") or [], draft)

    log.info(
        "Ricognizione su %.0fs: dominio='%s' confidenza=%s, %s termini (%s dopo il filtro)",
        sampled_s, inferred.get("domain"), inferred.get("confidence"),
        len(inferred.get("terms") or []), len(terms),
    )

    return {
        "ok": bool(inferred.get("topic") or terms),
        "sampled_s": round(sampled_s),
        "topic": inferred.get("topic", ""),
        "domain": inferred.get("domain", ""),
        "confidence": inferred.get("confidence", "bassa"),
        "terms": terms,
        "terms_discarded": max(0, len(inferred.get("terms") or []) - len(terms)),
        "draft_excerpt": draft[:400],
        "error": inferred.get("error"),
    }


def _run(
    audio_path: Path,
    device: str,
    progress: ProgressFn,
    notices: list[str],
    options: Any,
    skip_recon: bool = False,
) -> dict[str, Any]:
    from . import recon as recon_mod

    audio = whisperx.load_audio(str(audio_path))
    duration_s = len(audio) / 16000.0

    # 0) Ricognizione ------------------------------------------------------
    recon_report: dict[str, Any] = {"enabled": False}
    prompt_override = settings.asr_options(options.initial_prompt, options.hotwords)
    use_recon = settings.auto_context and options.auto_context and not skip_recon

    if skip_recon or not use_recon:
        recon_report = {
            "enabled": False,
            "ok": bool(options.initial_prompt or options.hotwords),
            "reason": "Contesto impostato manualmente per questa riunione."
            if options.initial_prompt or options.hotwords
            else "Ricognizione disattivata e nessun contesto indicato.",
            "topic": options.initial_prompt,
            "confidence": "manuale",
            "applied_prompt": options.initial_prompt,
            "applied_hotwords": options.hotwords,
        }
        log.info("Trascrizione con contesto della riunione (%s caratteri)",
                 len(options.initial_prompt))
    else:
        progress("scouting", 8, "Ricognizione: capisco di cosa si parla")
        try:
            recon_report = {"enabled": True, **reconnoitre(audio, duration_s, device)}
        except Exception as exc:
            log.warning("Ricognizione fallita: %s", exc)
            recon_report = {"enabled": True, "ok": False, "reason": str(exc)}

        # Quel che ha scritto l'utente per QUESTA riunione ha la precedenza;
        # la ricognizione integra, non sostituisce.
        merged_prompt = recon_mod.merge_context(
            recon_report.get("topic", "") if recon_report.get("ok") else "",
            options.initial_prompt,
        )
        merged_terms = recon_mod.merge_terms(
            recon_report.get("terms") or [] if recon_report.get("ok") else [],
            options.hotwords,
        )
        prompt_override = settings.asr_options(merged_prompt, merged_terms)
        recon_report["applied_prompt"] = merged_prompt
        recon_report["applied_hotwords"] = merged_terms

    # 1) Trascrizione ------------------------------------------------------
    progress("transcribing", 15, "Trascrizione dell'audio in corso")
    model = _cache.asr(device, options=prompt_override)
    result = model.transcribe(
        audio,
        batch_size=settings.whisper_batch_size,
        language=settings.language,
        # Avanzamento reale spezzone per spezzone: serve a mostrare una
        # percentuale onesta e, soprattutto, a dare al worker punti frequenti
        # in cui accorgersi di una richiesta di annullamento.
        progress_callback=_phase_progress(progress, "transcribing", 15, 45, "Trascrizione"),
    )
    language = result.get("language") or settings.language or "it"
    segments = result.get("segments", [])
    if not segments:
        raise RuntimeError("Nessun parlato riconosciuto nell'audio.")

    # 2) Allineamento word-level ------------------------------------------
    progress("aligning", 45, "Allineamento delle parole sui tempi")
    try:
        align_model, metadata = _cache.align(language, device)
        result = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        segments = result.get("segments", segments)
    except Exception as exc:
        # Alcune lingue non hanno modello di allineamento: proseguiamo comunque,
        # la diarizzazione userà i timestamp a livello di segmento.
        log.warning("Allineamento non riuscito (%s): proseguo senza word-timestamps", exc)
        result = {"segments": segments}

    # 3) Diarizzazione -----------------------------------------------------
    progress("diarizing", 65, "Identificazione degli interlocutori")
    result, diarization = diarize(
        audio, result, device, options.min_speakers, options.max_speakers, progress
    )
    segments = result.get("segments", segments)

    # 4) Sovrapposizioni ---------------------------------------------------
    overlaps = diarization.get("overlaps") or []
    stats = overlap.summarize(overlaps, duration_s) if overlaps else None

    if overlaps and options.separate_overlaps:
        worth_it = stats["percent"] >= settings.separation_min_overlap_pct
        if not worth_it:
            notices.append(
                f"Separazione delle voci saltata: solo {stats['percent']}% di parlato "
                "sovrapposto, il risultato sarebbe pressoché identico."
            )
        else:
            progress("separating", 72, "Separazione delle voci sovrapposte")
            separated, sep_notice = _transcribe_separated(
                audio_path, device, prompt_override, options
            )
            if separated:
                segments = _merge_separated(segments, separated, overlaps)
                stats["separated"] = True
            if sep_notice:
                notices.append(sep_notice)

    if overlaps:
        # I segmenti vanno SEMPRE marcati: è il campo `overlap` a dire
        # all'interfaccia dove più voci parlavano insieme. Prima, quando la
        # separazione riusciva, questa riga veniva saltata del tutto e nessun
        # segmento riceveva la marcatura: la trascrizione tornava a mostrare
        # gli interventi come se fossero stati detti uno dopo l'altro.
        #
        # Cambia solo la duplicazione. Senza separazione un tratto sovrapposto
        # produce una riga per voce con lo stesso testo: si sa chi c'era, ma il
        # testo resta quello che Whisper è riuscito a sentire. Con le tracce
        # separate ogni voce ha già il proprio testo, quindi si marca soltanto.
        segments = overlap.annotate(
            segments, overlaps, duplicate=not stats.get("separated")
        )
        diarization["overlap_stats"] = stats

    # Il merge di segmenti consecutivi ha senso solo se sappiamo che appartengono
    # alla stessa voce: senza diarizzazione produrrebbe blocchi indistinti.
    normalized = normalize_segments(segments, merge=diarization["ok"])
    if not normalized:
        raise RuntimeError("Trascrizione vuota dopo la normalizzazione.")

    return {
        "segments": normalized,
        "language": language,
        "duration_s": duration_s,
        "diarization": diarization,
        "notices": notices,
        "recon": recon_report,
    }


def _transcribe_separated(
    audio_path: Path, device: str, asr_options: dict | None, options: Any
) -> tuple[list[dict[str, Any]], str | None]:
    """Separa le voci e trascrive ogni traccia per conto suo.

    Il guadagno vero sta qui: ogni traccia contiene una sola voce, quindi
    Whisper lavora nelle condizioni per cui è stato addestrato e recupera
    anche il parlato che nella miscela veniva coperto.
    """
    from . import separation

    result = separation.separate(str(audio_path), audio_path.stem)
    if not result.get("ok"):
        return [], result.get("error") or "Separazione delle voci non riuscita."

    sources = result.get("sources") or []
    if not sources:
        return [], "La separazione non ha prodotto tracce utilizzabili."

    # Il modello di separazione ha appena rilasciato la GPU: ricarichiamo
    # Whisper con le stesse opzioni della trascrizione principale.
    out: list[dict[str, Any]] = []
    for source in sources:
        path = Path(source["path"])
        if not path.exists() or source.get("speech_s", 0) < 1.0:
            continue
        try:
            audio = whisperx.load_audio(str(path))
            model = _cache.asr(device, options=asr_options)
            res = model.transcribe(
                audio, batch_size=settings.whisper_batch_size, language=settings.language
            )
        except Exception as exc:
            log.warning("Traccia %s non trascritta: %s", source["speaker"], exc)
            continue

        # Allineamento anche qui. Senza, i segmenti della traccia separata non
        # hanno i tempi di parola, quindi non si possono tagliare sui confini
        # delle sovrapposizioni: restano interi anche di trenta secondi e
        # diventano inutilizzabili — o si scartano (e la separazione non serve
        # a niente) o si inseriscono interi (e duplicano la miscela).
        aligned = res.get("segments", [])
        try:
            align_model, metadata = _cache.align(settings.language or "it", device)
            aligned = whisperx.align(
                aligned, align_model, metadata, audio, device,
                return_char_alignments=False,
            ).get("segments", aligned)
        except Exception as exc:
            log.warning(
                "Traccia %s non allineata (%s): resterà tagliabile solo per segmento",
                source["speaker"], exc,
            )

        for seg in aligned:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            entry = {
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "speaker": source["speaker"],
                "text": text,
                "separated": True,
            }
            if seg.get("words"):
                entry["words"] = seg["words"]
            out.append(entry)
        # Le tracce separate pesano quanto l'originale: via appena usate.
        path.unlink(missing_ok=True)

    if not out:
        return [], "Le tracce separate non contenevano parlato riconoscibile."

    log.info("Trascrizione da tracce separate: %s segmenti da %s voci", len(out), len(sources))
    return out, None


def _merge_separated(
    base: list[dict[str, Any]],
    separated: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sostituisce i tratti sovrapposti con quelli ricavati dalle tracce separate.

    Fuori dalle sovrapposizioni la trascrizione della miscela è migliore: la
    separazione introduce artefatti che non servono dove c'è una voce sola.
    Quindi teniamo l'originale e rimpiazziamo solo dove serve davvero.

    Il "dove" va inteso alla lettera. La versione precedente decideva per
    segmento intero con un test di semplice contatto: un segmento di venti
    secondi che sfiorava per mezzo secondo una sovrapposizione veniva scartato
    tutto dalla base e reintrodotto tutto dalle tracce separate. Poiché i due
    insiemi hanno confini diversi, lo stesso parlato finiva due volte
    nell'elenco, con orari leggermente diversi e testo quasi identico: da qui
    le frasi ripetute. Su una riunione di prova erano 167 secondi coperti due
    volte su 679, un quarto del totale.

    Ora si taglia prima sui confini delle sovrapposizioni, poi ogni pezzo va
    da una parte sola — con una regola che rende impossibile perdere parlato:

    **un pezzo della base si scarta solo se qualcosa lo sostituisce davvero.**

    Serve perché le tracce separate non passano dall'allineamento e quindi non
    hanno i tempi di parola: non si possono tagliare, restano interi anche di
    trenta secondi, e non coprono ordinatamente le finestre di sovrapposizione.
    Scartare la base "perché lì ci pensano le tracce separate" senza verificare
    che sia vero lascia buchi di silenzio nella trascrizione — che è l'errore
    peggiore dei tre possibili: una ripetizione si legge, un buco no.
    """
    spans = overlap.overlap_spans(overlaps)
    base_pieces = overlap.split_at_boundaries(base, overlaps)
    sep_pieces = overlap.split_at_boundaries(separated, overlaps)

    # Dalle tracce separate si prende ciò che sta davvero dentro una
    # sovrapposizione, non ciò che la sfiora.
    replaced = [s for s in sep_pieces if overlap.fraction_inside(s, spans) >= 0.5]
    taken = overlap.merge_spans(
        [(s.get("start", 0.0), s.get("end", 0.0)) for s in replaced]
    )

    kept = []
    for piece in base_pieces:
        in_overlap = overlap.fraction_inside(piece, spans) >= 0.5
        # `taken` è ciò che le tracce separate coprono per davvero: se quel
        # tratto non è coperto, la base resta dov'è.
        substituted = overlap.fraction_inside(piece, taken) >= 0.5
        if not (in_overlap and substituted):
            kept.append(piece)

    result = sorted(kept + replaced, key=lambda s: (s.get("start", 0.0), s.get("speaker", "")))

    # Invariante esplicito: la fusione non può ridurre l'audio coperto da
    # testo. Se accade è un difetto, e va scritto nei log invece di arrivare
    # all'utente come una trascrizione che salta pezzi.
    before = overlap.covered_seconds([(s.get("start", 0.0), s.get("end", 0.0)) for s in base])
    after = overlap.covered_seconds([(s.get("start", 0.0), s.get("end", 0.0)) for s in result])
    # La soglia è proporzionale, non assoluta: tagliare sui confini di parola
    # costa qualche decimo di secondo per ogni confine, ed è normale. Quello
    # che va intercettato è il crollo — sul caso che ha prodotto questo
    # controllo la copertura era scesa del 36%.
    if after < before * 0.95 - 1.0:
        log.error(
            "Fusione delle tracce separate: copertura scesa da %.1fs a %.1fs "
            "(-%.0f%%). Tengo la trascrizione della miscela e uso le tracce "
            "separate solo in aggiunta: meglio qualche ripetizione che un buco.",
            before, after, (1 - after / before) * 100 if before else 0,
        )
        # Ripiego prudente: meglio qualche ripetizione che un buco.
        result = sorted(
            base_pieces + replaced,
            key=lambda s: (s.get("start", 0.0), s.get("speaker", "")),
        )

    log.info(
        "Fusione: %s pezzi dalla miscela, %s dalle tracce separate "
        "(base %s->%s, separate %s->%s) · copertura %.1fs -> %.1fs",
        len(kept), len(replaced), len(base), len(base_pieces),
        len(separated), len(sep_pieces), before, after,
    )
    return result


def normalize_segments(
    raw_segments: list[dict[str, Any]], merge: bool = True
) -> list[dict[str, Any]]:
    """Pulisce i segmenti e, se richiesto, fonde quelli contigui della stessa voce.

    Questa funzione ricostruisce ogni segmento campo per campo, quindi tutto
    ciò che non viene copiato esplicitamente qui sparisce prima del
    salvataggio. Vale in particolare per `words`: i tempi di parola sono
    prodotti dall'allineamento, usati per decidere i cambi di voce, e poi
    andavano perduti — l'interfaccia non poteva illuminare il parlato parola
    per parola perché il dato non arrivava mai al database.
    """
    cleaned: list[dict[str, Any]] = []
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker") or _majority_speaker(seg) or "SPEAKER_?"
        entry = {
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "speaker": str(speaker),
            "text": text,
        }
        if seg.get("overlap"):
            entry["overlap"] = list(seg["overlap"])
        if seg.get("separated"):
            entry["separated"] = True
        if seg.get("overlap_primary") is not None:
            entry["overlap_primary"] = bool(seg["overlap_primary"])

        # Solo le parole utili: quelle senza testo non servono a nessuno, e
        # senza istanti non sono illuminabili. Si tengono i quattro campi che
        # servono, non l'intero oggetto di WhisperX (che porta anche i
        # punteggi di allineamento, inutili qui e pesanti da trasferire).
        words = [
            {
                "word": (w.get("word") or "").strip(),
                "start": w.get("start"),
                "end": w.get("end"),
                "speaker": w.get("speaker"),
            }
            for w in (seg.get("words") or [])
            if (w.get("word") or "").strip()
        ]
        if words:
            entry["words"] = words
        cleaned.append(entry)

    if not merge:
        return cleaned

    merged: list[dict[str, Any]] = []
    for seg in cleaned:
        prev = merged[-1] if merged else None
        contiguous = (
            prev is not None
            and prev["speaker"] == seg["speaker"]
            and seg["speaker"] != "SPEAKER_?"
            and seg["start"] - prev["end"] <= 1.5
            and len(prev["text"]) < 600
            # Un tratto sovrapposto non va fuso con uno pulito: perderebbe
            # la marcatura e il lettore non saprebbe più dove non fidarsi.
            and bool(prev.get("overlap")) == bool(seg.get("overlap"))
        )
        if contiguous:
            prev["text"] = f"{prev['text']} {seg['text']}".strip()
            prev["end"] = seg["end"]
            # I tempi di parola vanno uniti insieme al testo: fondere le frasi
            # e lasciare indietro le parole produrrebbe un segmento in cui
            # l'evidenziazione si ferma a metà.
            if seg.get("words"):
                prev["words"] = (prev.get("words") or []) + seg["words"]
        else:
            merged.append(seg)
    return merged


def _majority_speaker(seg: dict[str, Any]) -> str | None:
    """Se il segmento non ha speaker, lo deduce dalle singole parole."""
    words = seg.get("words") or []
    counts: dict[str, int] = {}
    for word in words:
        spk = word.get("speaker")
        if spk:
            counts[spk] = counts.get(spk, 0) + 1
    return max(counts, key=counts.get) if counts else None
