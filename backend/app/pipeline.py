"""Worker: coda dei job in background, un job GPU alla volta."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

from . import store, summarize, transcribe
from .config import settings
from .models import JobStatus, Segment

log = logging.getLogger(__name__)

_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_stop = threading.Event()
_current_job: str | None = None

#: Contesto imposto a mano per la prossima esecuzione di un job specifico
#: (usato dalla ri-trascrizione con contesto corretto dall'utente).
_pending_context: dict[str, dict] = {}
_context_lock = threading.Lock()

#: Job di cui è stato chiesto l'annullamento.
_cancelled: set[str] = set()
_cancel_lock = threading.Lock()


class JobCancelled(Exception):
    """Sollevata dal callback di avanzamento quando l'utente annulla."""


# ---------------------------------------------------------------------------
def start_worker() -> None:
    global _worker
    if _worker and _worker.is_alive():
        return
    _stop.clear()
    _worker = threading.Thread(target=_loop, name="timbro-worker", daemon=True)
    _worker.start()
    log.info("Worker avviato")


def stop_worker() -> None:
    _stop.set()
    _queue.put("__stop__")


def enqueue(job_id: str, context: dict | None = None) -> None:
    if context is not None:
        with _context_lock:
            _pending_context[job_id] = context
    _queue.put(job_id)


def _context_of(job_id: str) -> dict | None:
    """Consuma il contesto imposto per questo job, se presente."""
    with _context_lock:
        return _pending_context.pop(job_id, None)


def queue_size() -> int:
    return _queue.qsize()


def is_busy() -> bool:
    """True se un job sta occupando la GPU o è in attesa in coda."""
    return _current_job is not None or not _queue.empty()


def current_job() -> str | None:
    return _current_job


# ---------------------------------------------------------------------------
# Annullamento
# ---------------------------------------------------------------------------
def request_cancel(job_id: str) -> str:
    """Chiede l'annullamento. Restituisce cosa è successo, per l'interfaccia.

    L'annullamento è cooperativo: non si può interrompere a metà un'inferenza
    su GPU senza lasciare la memoria in uno stato incerto. Il flag viene
    controllato a ogni avanzamento — che durante trascrizione e diarizzazione
    arriva a ogni spezzone — quindi in pratica si ferma entro pochi secondi.
    """
    with _cancel_lock:
        _cancelled.add(job_id)

    if _current_job == job_id:
        log.info("Annullamento richiesto per il job in corso %s", job_id)
        return "running"

    # Non ancora partito: lo segniamo subito come annullato, così l'interfaccia
    # non aspetta. Il flag resta però attivo: il job è ancora dentro la coda e
    # il worker deve trovarlo per scartarlo invece di elaborarlo.
    store.update_job(
        job_id,
        status=JobStatus.CANCELLED.value,
        progress=0,
        message="Annullato prima di iniziare",
        error=None,
    )
    log.info("Job %s annullato mentre era in coda", job_id)
    return "queued"


def is_cancelled(job_id: str) -> bool:
    with _cancel_lock:
        return job_id in _cancelled


def _clear_cancel(job_id: str) -> None:
    with _cancel_lock:
        _cancelled.discard(job_id)


# ---------------------------------------------------------------------------
def _loop() -> None:
    global _current_job
    while not _stop.is_set():
        job_id = _queue.get()
        if job_id == "__stop__":
            break
        _current_job = job_id
        try:
            # Può essere stato annullato mentre era in coda
            if is_cancelled(job_id):
                raise JobCancelled
            _process(job_id)
        except JobCancelled:
            log.info("Job %s annullato dall'utente", job_id)
            store.update_job(
                job_id,
                status=JobStatus.CANCELLED.value,
                message="Annullato",
                error=None,
            )
            transcribe.release_gpu()
        except Exception as exc:  # il worker non deve mai morire
            log.exception("Job %s fallito", job_id)
            store.update_job(
                job_id,
                status=JobStatus.ERROR.value,
                error=str(exc),
                message="Elaborazione fallita",
            )
        finally:
            _clear_cancel(job_id)
            _current_job = None
            _queue.task_done()


def _process(job_id: str) -> None:
    job = store.get_job(job_id)
    if job is None:
        log.warning("Job %s non trovato, ignoro", job_id)
        return

    audio_path = store.get_audio_path(job_id)
    if not audio_path or not Path(audio_path).exists():
        raise FileNotFoundError(f"File audio mancante per il job {job_id}")

    def progress(status: str, pct: int, message: str) -> None:
        # Punto di annullamento: viene attraversato a ogni spezzone di audio
        # durante le fasi lunghe, quindi la richiesta ha effetto in pochi secondi.
        if is_cancelled(job_id):
            raise JobCancelled
        store.update_job(job_id, status=status, progress=pct, message=message, error=None)

    # 0) Controllo preliminare: se la diarizzazione non può funzionare l'utente
    #    lo deve sapere ORA, non dopo dieci minuti di trascrizione.
    warnings = _precheck_diarization()
    if warnings:
        store.update_job(job_id, warnings=warnings)

    # 1) ASR + diarizzazione, con i parametri di QUESTA riunione
    override = _context_of(job_id)
    options = store.get_options(job_id)
    skip_recon = bool(override and override.get("skip_recon"))

    # Data della riunione: dai metadati dell'audio, con ripiego sulla data del
    # file. Senza, «entro domattina» resta irrisolvibile. Non si sovrascrive
    # una data già corretta a mano.
    job_now = store.get_job(job_id)
    if not (job_now and job_now.recorded_at):
        detected = transcribe.probe_recorded_at(Path(audio_path))
        store.update_job(job_id, recorded_at=detected or (job_now.created_at if job_now else ""))

    progress(JobStatus.TRANSCRIBING.value, 5, "Preparazione dell'audio")
    result = transcribe.transcribe(
        Path(audio_path), progress, options=options, skip_recon=skip_recon
    )
    segments = [Segment(**s) for s in result["segments"]]

    diarization = result.get("diarization") or {}
    warnings = _diarization_warnings(diarization) + list(result.get("notices") or [])

    store.update_job(
        job_id,
        segments=[s.model_dump() for s in segments],
        duration_s=result["duration_s"],
        language=result["language"],
        warnings=warnings,
        recon=result.get("recon"),
        overlap=diarization.get("overlap_stats"),
    )

    # 2) Libera la VRAM prima di chiamare l'LLM: condividono la stessa GPU
    transcribe.release_gpu()

    # 3) Analisi LLM
    progress(JobStatus.SUMMARIZING.value, 75, "Analisi della trascrizione")
    analysis = summarize.analyze(segments, progress)
    # Le scadenze dette a voce diventano date vere, ancorate
    # all'istante della riunione. L'aritmetica sta in `dates`,
    # non nel prompt: un modello che conta i giorni sbaglia.
    _job = store.get_job(job_id)
    analysis = summarize.resolve_due_dates(
        analysis, (_job.recorded_at if _job else '') or (_job.created_at if _job else '')
    )

    store.update_job(
        job_id,
        status=JobStatus.DONE.value,
        progress=100,
        message="Completato",
        analysis=analysis.model_dump(),
        error=None,
    )
    # Indicizza per la ricerca d'archivio. Non solleva mai: una
    # riunione elaborata bene non deve fallire per l'indice.
    store.reindex_job(job_id)
    # Gli impegni presi diventano task veri. Non tocca quelli
    # esistenti: commenti e riprogrammazioni sono lavoro umano.
    store.sync_tasks_from_job(job_id)

    if settings.delete_audio_after:
        Path(audio_path).unlink(missing_ok=True)

    log.info("Job %s completato (%s segmenti)", job_id, len(segments))


# ---------------------------------------------------------------------------
# Avvisi sulla diarizzazione
# ---------------------------------------------------------------------------
def _precheck_diarization() -> list[str]:
    """Verifica prima della trascrizione che la diarizzazione sia praticabile."""
    if not settings.hf_token:
        return [
            "Nessun token Hugging Face configurato: la trascrizione verrà prodotta "
            "senza distinguere gli interlocutori. Inseriscilo in ⚙ Impostazioni."
        ]
    return []


def _diarization_warnings(diarization: dict) -> list[str]:
    """Traduce il report della diarizzazione in avvisi per l'interfaccia."""
    if diarization.get("ok"):
        if diarization.get("speakers", 0) == 1:
            return [
                "È stata rilevata una sola voce. Se la riunione aveva più "
                "partecipanti, imposta interlocutori minimi e massimi in "
                "⚙ Impostazioni e rilancia il riconoscimento voci."
            ]
        return []

    detail = diarization.get("error") or "causa sconosciuta"
    return [
        f"Riconoscimento voci non riuscito: {detail} "
        "La trascrizione è comunque completa, ma senza etichette per chi parla. "
        "Risolto il problema puoi rifare solo questa fase, senza ri-trascrivere."
    ]


def rediarize(job_id: str) -> dict:
    """Rifà solo la diarizzazione su un job già trascritto.

    L'ASR è la parte lenta: dopo aver sistemato token o licenza non ha senso
    ripeterla. Riusiamo l'audio originale e i segmenti già prodotti.
    """
    job = store.get_job(job_id)
    if job is None:
        raise ValueError("Job inesistente")
    if not job.segments:
        raise ValueError("Il job non ha una trascrizione da rietichettare")

    audio_path = store.get_audio_path(job_id)
    if not audio_path or not Path(audio_path).exists():
        raise ValueError(
            "L'audio originale non è più disponibile: ricarica il file. "
            "(Disattiva 'Elimina il file audio dopo l'elaborazione' in ⚙ Impostazioni.)"
        )

    def progress(status: str, pct: int, message: str) -> None:
        store.update_job(job_id, status=status, progress=pct, message=message, error=None)

    progress(JobStatus.DIARIZING.value, 20, "Identificazione degli interlocutori")

    import whisperx  # import locale: pesante, serve solo qui

    audio = whisperx.load_audio(audio_path)
    result = {"segments": [s.model_dump() for s in job.segments]}
    opts = store.get_options(job_id).normalized()
    result, diarization = transcribe.diarize(
        audio, result, transcribe.pick_device(),
        opts.min_speakers, opts.max_speakers, progress,
    )

    segments = transcribe.normalize_segments(
        result.get("segments", []), merge=diarization["ok"]
    )
    warnings = _diarization_warnings(diarization)
    store.update_job(job_id, segments=segments, warnings=warnings)
    transcribe.release_gpu()

    if not diarization["ok"]:
        store.update_job(
            job_id,
            status=JobStatus.DONE.value,
            progress=100,
            message="Riconoscimento voci non riuscito",
        )
        return diarization

    # Con i nuovi speaker il verbale precedente non è più coerente: lo rifacciamo.
    progress(JobStatus.SUMMARIZING.value, 70, "Aggiornamento del verbale")
    analysis = summarize.analyze(
        [Segment(**s) for s in segments], progress, job.speaker_names
    )
    # Le scadenze dette a voce diventano date vere, ancorate
    # all'istante della riunione. L'aritmetica sta in `dates`,
    # non nel prompt: un modello che conta i giorni sbaglia.
    _job = store.get_job(job_id)
    analysis = summarize.resolve_due_dates(
        analysis, (_job.recorded_at if _job else '') or (_job.created_at if _job else '')
    )
    store.update_job(
        job_id,
        status=JobStatus.DONE.value,
        progress=100,
        message="Completato",
        analysis=analysis.model_dump(),
        error=None,
    )
    # Indicizza per la ricerca d'archivio. Non solleva mai: una
    # riunione elaborata bene non deve fallire per l'indice.
    store.reindex_job(job_id)
    # Gli impegni presi diventano task veri. Non tocca quelli
    # esistenti: commenti e riprogrammazioni sono lavoro umano.
    store.sync_tasks_from_job(job_id)
    return diarization


# ---------------------------------------------------------------------------
def regenerate_analysis(job_id: str) -> None:
    """Rigenera solo il verbale (senza ri-trascrivere), es. dopo la rinomina speaker."""
    job = store.get_job(job_id)
    if job is None or not job.segments:
        raise ValueError("Job inesistente o privo di trascrizione")

    def progress(status: str, pct: int, message: str) -> None:
        store.update_job(job_id, status=status, progress=pct, message=message, error=None)

    progress(JobStatus.SUMMARIZING.value, 75, "Rigenerazione del verbale")
    analysis = summarize.analyze(job.segments, progress, job.speaker_names)
    # Le scadenze dette a voce diventano date vere, ancorate
    # all'istante della riunione. L'aritmetica sta in `dates`,
    # non nel prompt: un modello che conta i giorni sbaglia.
    _job = store.get_job(job_id)
    analysis = summarize.resolve_due_dates(
        analysis, (_job.recorded_at if _job else '') or (_job.created_at if _job else '')
    )
    store.update_job(
        job_id,
        status=JobStatus.DONE.value,
        progress=100,
        message="Completato",
        analysis=analysis.model_dump(),
        error=None,
    )
    # Indicizza per la ricerca d'archivio. Non solleva mai: una
    # riunione elaborata bene non deve fallire per l'indice.
    store.reindex_job(job_id)
    # Gli impegni presi diventano task veri. Non tocca quelli
    # esistenti: commenti e riprogrammazioni sono lavoro umano.
    store.sync_tasks_from_job(job_id)
