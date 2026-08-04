"""Timbro - API FastAPI + serving della web UI."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import (
    export,
    gpu,
    hf,
    ollama_admin,
    pipeline,
    separation,
    settings_store,
    store,
    summarize,
    transcribe,
)
from .config import settings
from .models import (
    AppSettings,
    JobDetail,
    JobOptions,
    JobStatus,
    JobSummary,
    PullRequest,
    RenameRequest,
    RetranscribeRequest,
    SettingsOptions,
    SettingsPatch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("timbro")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

ALLOWED_SUFFIXES = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".wma", ".aac",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    settings_store.load_and_apply()
    store.init_db()
    recovered = store.recover_interrupted()
    if recovered:
        log.warning("%s job interrotti marcati come errore", recovered)
    pipeline.start_worker()
    log.info(
        "Timbro pronto | whisper=%s | llm=%s | device=%s",
        settings.whisper_model,
        settings.ollama_model,
        transcribe.pick_device(),
    )
    yield
    pipeline.stop_worker()


app = FastAPI(title="Timbro", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": transcribe.pick_device(),
        "whisper_model": settings.whisper_model,
        "diarization_model": settings.diarization_model,
        "hf_token_set": bool(settings.hf_token),
        "queue_size": pipeline.queue_size(),
        "busy": pipeline.is_busy(),
        "ollama": summarize.check_ollama(),
        "gpu": gpu.stats(),
        "separator": separation.available(),
    }


@app.get("/api/gpu")
def gpu_stats() -> dict:
    """Occupazione VRAM dell'intera scheda, aggiornabile di frequente."""
    return {**gpu.stats(), "busy": pipeline.is_busy()}


# ---------------------------------------------------------------------------
# Impostazioni
# ---------------------------------------------------------------------------
@app.get("/api/settings", response_model=AppSettings)
def read_settings() -> AppSettings:
    return settings_store.current()


@app.put("/api/settings", response_model=AppSettings)
def write_settings(patch: SettingsPatch) -> AppSettings:
    """Aggiorna solo i campi effettivamente inviati e li salva su disco."""
    if pipeline.is_busy():
        raise HTTPException(
            409,
            "Elaborazione in corso: attendi la fine del job prima di cambiare le impostazioni.",
        )
    try:
        current, reloaded = settings_store.update(patch.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if reloaded:
        log.info("Impostazioni modelli cambiate: ricarica al prossimo job")
    return current


@app.get("/api/settings/options", response_model=SettingsOptions)
def settings_options() -> SettingsOptions:
    """Valori proposti nelle tendine dell'interfaccia."""
    return SettingsOptions(
        whisper_models=[
            {"name": "tiny", "vram_gb": 1, "note": "Velocissimo, qualità bassa. Solo per prove."},
            {"name": "base", "vram_gb": 1, "note": "Molto veloce, qualità limitata."},
            {"name": "small", "vram_gb": 2, "note": "Buon compromesso su GPU piccole."},
            {"name": "medium", "vram_gb": 5, "note": "Qualità buona, velocità discreta."},
            {"name": "large-v3-turbo", "vram_gb": 6,
             "note": "Consigliato: qualità quasi da large-v3, 4x più veloce."},
            {"name": "large-v2", "vram_gb": 10, "note": "Ottimo su audio rumoroso."},
            {"name": "large-v3", "vram_gb": 10, "note": "Qualità massima, più lento."},
        ],
        compute_types=[
            {"name": "float16", "note": "Default su GPU moderne."},
            {"name": "int8_float16", "note": "Circa 40% di VRAM in meno, qualità quasi identica."},
            {"name": "bfloat16", "note": "Per Ampere e successive."},
            {"name": "float32", "note": "Massima precisione, molta VRAM."},
            {"name": "int8", "note": "Necessario se giri su CPU."},
        ],
        languages=[
            {"code": "it", "name": "Italiano"},
            {"code": "en", "name": "Inglese"},
            {"code": "fr", "name": "Francese"},
            {"code": "de", "name": "Tedesco"},
            {"code": "es", "name": "Spagnolo"},
            {"code": "pt", "name": "Portoghese"},
            {"code": "nl", "name": "Olandese"},
            {"code": "auto", "name": "Rilevamento automatico"},
        ],
        diarization_models=[
            "pyannote/speaker-diarization-community-1",
            "pyannote/speaker-diarization-3.1",
        ],
        suggested_llms=ollama_admin.SUGGESTED_LLMS,
        installed_llms=ollama_admin.installed_names(),
    )


@app.post("/api/settings/verify-hf")
def verify_hf(payload: dict | None = None) -> dict:
    """Verifica token e accesso ai repo gated. Se `token` è assente usa quello salvato."""
    token = (payload or {}).get("token") or None
    return hf.verify(token)


@app.post("/api/settings/diagnose-separation")
def diagnose_separation() -> dict:
    """Verifica il servizio di separazione: raggiungibile, versione, licenze."""
    if pipeline.is_busy():
        raise HTTPException(409, "Elaborazione in corso: riprova a fine job.")
    return separation.diagnose()


@app.post("/api/settings/diagnose-diarization")
def diagnose_diarization() -> dict:
    """Istanzia davvero la pipeline di diarizzazione e riporta l'errore esatto.

    È il controllo che distingue "token sbagliato" da "licenza non accettata"
    da "incompatibilità di libreria", invece di lasciare l'utente davanti a
    una trascrizione senza etichette.
    """
    if pipeline.is_busy():
        raise HTTPException(409, "Elaborazione in corso: riprova a fine job.")
    return transcribe.diagnose()


@app.get("/api/settings/llms")
def list_llms() -> dict:
    try:
        return {"models": ollama_admin.list_models(), "reachable": True}
    except Exception as exc:
        return {"models": [], "reachable": False, "error": str(exc)}


@app.post("/api/settings/llms/pull", status_code=202)
def pull_llm(payload: PullRequest) -> dict:
    try:
        return ollama_admin.start_pull(payload.model)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/settings/llms/pull")
def pull_llm_status() -> dict:
    return ollama_admin.pull_status()


# `response_model=None` è necessario: con `from __future__ import annotations`
# FastAPI risolve `-> None` in `NoneType` e lo scambia per un corpo di risposta,
# vietato con 204. Vale per tutte le rotte 204 di questo file.
@app.delete("/api/settings/llms/{model:path}", status_code=204, response_model=None)
def remove_llm(model: str) -> None:
    try:
        ollama_admin.delete_model(model)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Rimozione fallita: {exc}") from exc


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------
@app.post("/api/jobs", response_model=JobSummary, status_code=201)
async def create_job(
    file: UploadFile,
    initial_prompt: str = Form(default=""),
    hotwords: str = Form(default=""),
    min_speakers: str = Form(default=""),
    max_speakers: str = Form(default=""),
    auto_context: bool = Form(default=True),
    separate_overlaps: bool = Form(default=False),
) -> JobSummary:
    """Carica un audio con i parametri di QUESTA riunione.

    Argomento, termini e numero di partecipanti cambiano da un incontro
    all'altro: arrivano col file, non dalla configurazione globale.
    """
    try:
        options = JobOptions(
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            min_speakers=int(min_speakers) if min_speakers.strip() else None,
            max_speakers=int(max_speakers) if max_speakers.strip() else None,
            auto_context=auto_context,
            separate_overlaps=separate_overlaps,
        ).normalized()
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, f"Parametri della riunione non validi: {exc}") from exc

    original = Path(file.filename or "audio").name
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            415,
            f"Formato '{suffix or 'sconosciuto'}' non supportato. "
            f"Ammessi: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    job_id = uuid.uuid4().hex[:12]
    settings.ensure_dirs()
    dest = settings.uploads_dir / f"{job_id}{suffix}"

    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, f"File oltre il limite di {settings.max_upload_mb} MB")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "File vuoto")

    store.create_job(job_id, original, str(dest), options)
    pipeline.enqueue(job_id)
    log.info(
        "Job %s creato per '%s' (%.1f MB) | contesto=%s termini=%s voci=%s-%s",
        job_id, original, written / 1e6,
        "sì" if options.initial_prompt else "no",
        "sì" if options.hotwords else "no",
        options.min_speakers or "auto", options.max_speakers or "auto",
    )

    job = store.get_job(job_id)
    assert job is not None
    return JobSummary(
        **job.model_dump(
            exclude={"segments", "analysis", "speaker_names", "recon", "options"}
        )
    )


@app.get("/api/jobs", response_model=list[JobSummary])
def list_jobs(limit: int = 100) -> list[JobSummary]:
    return store.list_jobs(limit=limit)


@app.get("/api/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str) -> JobDetail:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job non trovato")
    return job


@app.delete("/api/jobs/{job_id}", status_code=204, response_model=None)
def delete_job(job_id: str) -> None:
    audio_path = store.get_audio_path(job_id)
    if not store.delete_job(job_id):
        raise HTTPException(404, "Job non trovato")
    if audio_path:
        Path(audio_path).unlink(missing_ok=True)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Annulla un'elaborazione in corso o in coda.

    L'annullamento è cooperativo: il worker se ne accorge al primo punto di
    avanzamento utile, che durante le fasi lunghe arriva a ogni spezzone.
    """
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job non trovato")
    if job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
        raise HTTPException(409, "Il job non è in elaborazione")

    where = pipeline.request_cancel(job_id)
    return {
        "cancelled": True,
        "was": where,
        "message": "Annullato." if where == "queued"
        else "Annullamento richiesto: si ferma entro pochi secondi.",
    }


@app.post("/api/jobs/{job_id}/retry", response_model=JobSummary)
def retry_job(job_id: str) -> JobSummary:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job non trovato")
    audio_path = store.get_audio_path(job_id)
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(409, "Audio originale non più disponibile: ricarica il file")

    store.update_job(
        job_id,
        status=JobStatus.QUEUED.value,
        progress=0,
        message="In coda",
        error=None,
    )
    pipeline.enqueue(job_id)
    job = store.get_job(job_id)
    assert job is not None
    return JobSummary(
        **job.model_dump(
            exclude={"segments", "analysis", "speaker_names", "recon", "options"}
        )
    )


@app.post("/api/jobs/{job_id}/retranscribe", response_model=JobSummary)
def retranscribe_job(job_id: str, payload: RetranscribeRequest) -> JobSummary:
    """Rimette in coda il job usando il contesto corretto a mano.

    Serve dopo aver visto cosa ha dedotto la ricognizione: se il tema è giusto
    ma qualche termine è sbagliato, lo si corregge e si rilancia.
    """
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job non trovato")

    audio_path = store.get_audio_path(job_id)
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(
            409,
            "L'audio originale non è più disponibile: ricarica il file. "
            "(Disattiva 'Elimina il file audio dopo l'elaborazione' in ⚙ Impostazioni.)",
        )

    options = JobOptions(**payload.model_dump(exclude={"skip_recon"})).normalized()
    store.update_job(
        job_id,
        status=JobStatus.QUEUED.value,
        progress=0,
        message="In coda con i parametri corretti",
        error=None,
        warnings=[],
        options=options.model_dump(),
    )
    pipeline.enqueue(job_id, context={"skip_recon": payload.skip_recon})
    log.info("Job %s rimesso in coda con contesto manuale", job_id)

    job = store.get_job(job_id)
    assert job is not None
    return JobSummary(
        **job.model_dump(
            exclude={"segments", "analysis", "speaker_names", "recon", "options"}
        )
    )


@app.post("/api/jobs/{job_id}/rediarize", response_model=JobDetail)
def rediarize_job(job_id: str) -> JobDetail:
    """Rifà solo il riconoscimento voci, riusando la trascrizione esistente."""
    if pipeline.is_busy():
        raise HTTPException(409, "Elaborazione in corso: attendi la fine del job.")
    try:
        pipeline.rediarize(job_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        log.exception("Ri-diarizzazione fallita per %s", job_id)
        store.update_job(job_id, status=JobStatus.ERROR.value, error=str(exc))
        raise HTTPException(500, f"Ri-diarizzazione fallita: {exc}") from exc

    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job non trovato")
    return job


@app.post("/api/jobs/{job_id}/speakers", response_model=JobDetail)
def rename_speakers(job_id: str, payload: RenameRequest) -> JobDetail:
    """Assegna nomi reali agli speaker; opzionalmente rigenera il verbale."""
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job non trovato")
    if job.status in (JobStatus.QUEUED, JobStatus.TRANSCRIBING, JobStatus.ALIGNING,
                      JobStatus.DIARIZING, JobStatus.SUMMARIZING):
        raise HTTPException(409, "Job ancora in elaborazione")

    names = {k: v.strip() for k, v in payload.speaker_names.items() if v.strip()}
    store.update_job(job_id, speaker_names=names)

    if payload.regenerate and job.segments:
        try:
            pipeline.regenerate_analysis(job_id)
        except Exception as exc:
            log.exception("Rigenerazione fallita per %s", job_id)
            store.update_job(job_id, status=JobStatus.ERROR.value, error=str(exc))
            raise HTTPException(500, f"Rigenerazione fallita: {exc}") from exc

    updated = store.get_job(job_id)
    assert updated is not None
    return updated


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@app.get("/api/jobs/{job_id}/markdown", response_class=PlainTextResponse)
def markdown(job_id: str, transcript: bool = True, download: bool = False):
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job non trovato")
    content = export.to_markdown(job, include_transcript=transcript)

    headers = {}
    if download:
        stem = Path(job.filename).stem or job.id
        safe = "".join(c for c in stem if c.isalnum() or c in " -_").strip() or job.id
        headers["Content-Disposition"] = f'attachment; filename="{safe}.md"'
    return PlainTextResponse(content, media_type="text/markdown; charset=utf-8", headers=headers)


@app.get("/api/jobs/{job_id}/audio")
def audio(job_id: str):
    path = store.get_audio_path(job_id)
    if not path or not Path(path).exists():
        raise HTTPException(404, "Audio non disponibile")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Timbro</h1><p>Frontend non trovato. API su /docs</p>")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))
