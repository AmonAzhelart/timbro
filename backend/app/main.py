"""Timbro - API FastAPI + serving della web UI."""

from __future__ import annotations

import logging
import mimetypes
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Response, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import (
    dates,
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
from . import tasks as tasks_mod
from .config import settings
from .models import (
    AppSettings,
    Folder,
    FolderPatch,
    FolderRequest,
    Glossary,
    JobDetail,
    JobFiling,
    JobOptions,
    JobStatus,
    JobSummary,
    KnownSpeaker,
    MeetingAnalysis,
    OpenActionPoint,
    PullRequest,
    RecordedAtRequest,
    RenameRequest,
    RetranscribeRequest,
    SearchHit,
    SearchResponse,
    SettingsOptions,
    SettingsPatch,
    Tag,
    TagRequest,
    Task,
    TaskComment,
    TaskCreate,
    TaskPatch,
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

# Con i tempi per parola il dettaglio di una riunione lunga passa da qualche
# centinaio di kB a qualche MB, ed è JSON: si comprime di circa dieci volte.
# `minimum_size` evita di pagare la compressione sulle risposte piccole, che
# sono la maggior parte (stato, avanzamento, impostazioni).
app.add_middleware(GZipMiddleware, minimum_size=1024)


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
    store.remember_speakers(list(names.values()))

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


@app.patch("/api/jobs/{job_id}/analysis", response_model=JobDetail)
def edit_analysis(job_id: str, payload: MeetingAnalysis) -> JobDetail:
    """Sostituisce il verbale con la versione corretta a mano.

    Non rigenera nulla e non tocca la trascrizione: l'LLM sbaglia un nome o una
    scadenza e finora l'unico rimedio era rifare tutto, perdendo anche le
    correzioni giuste. Qui vince quello che scrive la persona.
    """
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job non trovato")
    if job.status not in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
        raise HTTPException(409, "Job ancora in elaborazione")

    store.update_job(job_id, analysis=payload.model_dump())
    # L'indice segue il verbale: senza, la ricerca continuerebbe a restituire
    # il testo vecchio, che è peggio di non trovare nulla.
    store.reindex_job(job_id)
    updated = store.get_job(job_id)
    assert updated is not None
    return updated


# ---------------------------------------------------------------------------
# Archivio: cartelle, etichette, collocazione
# ---------------------------------------------------------------------------
@app.get("/api/folders", response_model=list[Folder])
def folders() -> list[Folder]:
    return [Folder.model_validate(f) for f in store.list_folders()]


@app.post("/api/folders", response_model=Folder, status_code=201)
def add_folder(payload: FolderRequest) -> Folder:
    if payload.parent_id and not any(
        f["id"] == payload.parent_id for f in store.list_folders()
    ):
        raise HTTPException(404, "Cartella superiore non trovata")
    return Folder.model_validate(
        store.create_folder(payload.name, payload.parent_id, payload.color)
    )


@app.patch("/api/folders/{folder_id}", response_model=list[Folder])
def edit_folder(folder_id: str, payload: FolderPatch) -> list[Folder]:
    error = store.update_folder(
        folder_id, name=payload.name, parent_id=payload.parent_id,
        color=payload.color, move=payload.move,
    )
    if error:
        raise HTTPException(400, error)
    return [Folder.model_validate(f) for f in store.list_folders()]


@app.delete("/api/folders/{folder_id}", status_code=204, response_model=None)
def remove_folder(folder_id: str):
    """Elimina la cartella e le sue sottocartelle; le riunioni tornano alla radice."""
    if not store.delete_folder(folder_id):
        raise HTTPException(404, "Cartella non trovata")
    return Response(status_code=204)


@app.get("/api/tags", response_model=list[Tag])
def tags() -> list[Tag]:
    return [Tag.model_validate(t) for t in store.list_tags()]


@app.post("/api/tags", response_model=Tag, status_code=201)
def add_tag(payload: TagRequest) -> Tag:
    return Tag.model_validate({**store.ensure_tag(payload.name, payload.color), "jobs": 0})


@app.delete("/api/tags/{tag_id}", status_code=204, response_model=None)
def remove_tag(tag_id: str):
    if not store.delete_tag(tag_id):
        raise HTTPException(404, "Etichetta non trovata")
    return Response(status_code=204)


@app.post("/api/jobs/{job_id}/recorded-at", response_model=JobDetail)
def set_recorded_at(job_id: str, payload: RecordedAtRequest) -> JobDetail:
    """Corregge la data della riunione e ricalcola le scadenze.

    La data viene dedotta dai metadati dell'audio, che però non sempre ci
    sono e non sempre sono giusti. Correggendola qui, «entro domattina» si
    sposta sul giorno corretto senza dover ri-trascrivere nulla.
    """
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job non trovato")
    try:
        moment = datetime.fromisoformat(payload.recorded_at)
    except ValueError as exc:
        raise HTTPException(422, "Data non valida: usa il formato AAAA-MM-GGTHH:MM") from exc

    store.update_job(job_id, recorded_at=moment.isoformat(timespec="seconds"))

    if job.analysis:
        updated_analysis = summarize.resolve_due_dates(job.analysis, moment.isoformat())
        store.update_job(job_id, analysis=updated_analysis.model_dump())
        store.reindex_job(job_id, with_vectors=False)

    updated = store.get_job(job_id)
    assert updated is not None
    return updated


@app.post("/api/jobs/{job_id}/filing", response_model=JobDetail)
def file_job(job_id: str, payload: JobFiling) -> JobDetail:
    """Sposta una riunione in una cartella e ne aggiorna le etichette."""
    if store.get_job(job_id) is None:
        raise HTTPException(404, "Job non trovato")
    if payload.move:
        store.set_job_folder(job_id, payload.folder_id)
    if payload.tags is not None:
        store.set_job_tags(job_id, payload.tags)
    updated = store.get_job(job_id)
    assert updated is not None
    return updated


# ---------------------------------------------------------------------------
# Ricerca su tutto l'archivio
# ---------------------------------------------------------------------------
@app.get("/api/search", response_model=SearchResponse)
def search_archive(q: str = "", limit: int = 40, kinds: str = "") -> SearchResponse:
    wanted = [k for k in kinds.split(",") if k.strip()] or None
    result = store.search_archive(q, limit=max(1, min(limit, 100)), kinds=wanted)
    return SearchResponse(
        hits=[SearchHit.model_validate(h) for h in result["hits"]],
        semantic=result["semantic"],
        note=result["note"],
    )


@app.post("/api/search/reindex")
def rebuild_index(vectors: bool = True) -> dict:
    """Ricostruisce l'indice su tutte le riunioni già elaborate.

    Non è automatico all'avvio: su un archivio grande sarebbe un costo a
    sorpresa a ogni riavvio del container.
    """
    return store.reindex_all(with_vectors=vectors)


# ---------------------------------------------------------------------------
# Impegni, glossari, rubrica delle voci
# ---------------------------------------------------------------------------
@app.get("/api/action-points", response_model=list[OpenActionPoint])
def action_points(include_done: bool = False) -> list[OpenActionPoint]:
    return [OpenActionPoint.model_validate(p) for p in store.all_action_points(include_done)]


@app.post("/api/jobs/{job_id}/action-points/{index}")
def toggle_action_point(job_id: str, index: int, done: bool = True) -> dict:
    if not store.set_action_done(job_id, index, done):
        raise HTTPException(404, "Action point non trovato")
    store.reindex_job(job_id, with_vectors=False)
    return {"ok": True, "done": done}


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
@app.get("/api/tasks", response_model=list[Task])
def tasks(include_done: bool = True) -> list[Task]:
    return [Task.model_validate(t) for t in store.list_tasks(include_done)]


@app.post("/api/tasks", response_model=Task, status_code=201)
def add_task(payload: TaskCreate) -> Task:
    data = payload.model_dump()
    if data["status"] not in tasks_mod.STATI:
        raise HTTPException(422, f"Stato non valido: {data['status']}")
    created = store.create_task(data)
    return Task.model_validate(created)


@app.patch("/api/tasks/{task_id}", response_model=Task)
def edit_task(task_id: str, payload: TaskPatch) -> Task:
    before = store.get_task(task_id)
    if before is None:
        raise HTTPException(404, "Task non trovato")

    fields = payload.model_dump(exclude_unset=True)
    reason = fields.pop("reason", None)
    if fields.get("status") and fields["status"] not in tasks_mod.STATI:
        raise HTTPException(422, f"Stato non valido: {fields['status']}")

    updated = store.update_task(task_id, fields)
    if updated is None:
        raise HTTPException(404, "Task non trovato")

    # Uno spostamento di scadenza e un cambio di stato finiscono nello storico
    # da soli: ricostruire fra un mese "quando è slittato e perché" senza
    # traccia automatica è impossibile, e nessuno annota a mano.
    if "due_at" in fields and fields["due_at"] != before["due_at"]:
        prima = dates.describe(before["due_at"], before["due_precision"])
        dopo = dates.describe(updated["due_at"], updated["due_precision"])
        store.add_comment(
            task_id,
            f"Scadenza spostata da {prima} a {dopo}" + (f" — {reason}" if reason else ""),
            kind="sistema",
        )
    elif "status" in fields and fields["status"] != before["status"]:
        etichetta = tasks_mod.STATI_LABEL.get(fields["status"], fields["status"])
        store.add_comment(
            task_id,
            f"Stato: {etichetta}" + (f" — {reason}" if reason else ""),
            kind="sistema",
        )
    elif reason:
        store.add_comment(task_id, reason, kind="commento")

    return Task.model_validate({**updated, "comments": len(store.task_comments(task_id))})


@app.delete("/api/tasks/{task_id}", status_code=204, response_model=None)
def remove_task(task_id: str):
    if not store.delete_task(task_id):
        raise HTTPException(404, "Task non trovato")
    return Response(status_code=204)


@app.get("/api/tasks/{task_id}/comments", response_model=list[TaskComment])
def comments(task_id: str) -> list[TaskComment]:
    if store.get_task(task_id) is None:
        raise HTTPException(404, "Task non trovato")
    return [TaskComment.model_validate(c) for c in store.task_comments(task_id)]


@app.post("/api/tasks/{task_id}/comments", response_model=TaskComment, status_code=201)
def add_task_comment(task_id: str, payload: TaskComment) -> TaskComment:
    if store.get_task(task_id) is None:
        raise HTTPException(404, "Task non trovato")
    return TaskComment.model_validate(store.add_comment(task_id, payload.body))


@app.post("/api/jobs/{job_id}/tasks")
def sync_tasks(job_id: str) -> dict:
    """Crea i task mancanti dagli action point di questa riunione."""
    if store.get_job(job_id) is None:
        raise HTTPException(404, "Job non trovato")
    return {"created": store.sync_tasks_from_job(job_id)}


@app.get("/api/glossaries", response_model=list[Glossary])
def glossaries() -> list[Glossary]:
    return [Glossary.model_validate(g) for g in store.list_glossaries()]


@app.post("/api/glossaries", response_model=Glossary)
def save_glossary(payload: Glossary) -> Glossary:
    return Glossary.model_validate(
        store.save_glossary(
            payload.name, payload.prompt, payload.hotwords,
            payload.folder_id, payload.id or None,
        )
    )


@app.delete("/api/glossaries/{glossary_id}", status_code=204, response_model=None)
def remove_glossary(glossary_id: str):
    if not store.delete_glossary(glossary_id):
        raise HTTPException(404, "Glossario non trovato")
    return Response(status_code=204)


@app.get("/api/speakers", response_model=list[KnownSpeaker])
def speakers() -> list[KnownSpeaker]:
    return [KnownSpeaker.model_validate(s) for s in store.known_speakers()]


@app.delete("/api/speakers/{name}", status_code=204, response_model=None)
def remove_speaker(name: str):
    if not store.forget_speaker(name):
        raise HTTPException(404, "Nome non trovato")
    return Response(status_code=204)


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


#: Il tipo MIME va dichiarato esplicitamente: senza, Starlette lo indovina
#: dall'estensione e su un fallimento ripiega su text/plain, che il tag
#: <audio> rifiuta di riprodurre. Le chiavi non coperte da `mimetypes`
#: (o coperte male, come .m4a) stanno qui.
_AUDIO_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


@app.get("/api/jobs/{job_id}/audio")
def audio(job_id: str):
    """Serve l'audio originale del job, per la riproduzione sincronizzata.

    `FileResponse` gestisce da sé le richieste Range, indispensabili perché il
    player possa saltare a un istante preciso senza scaricare tutto il file.
    """
    path = store.get_audio_path(job_id)
    if not path or not Path(path).exists():
        raise HTTPException(404, "Audio non disponibile")

    suffix = Path(path).suffix.lower()
    media_type = _AUDIO_TYPES.get(suffix) or mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(
        path,
        media_type=media_type,
        # Il file non cambia mai: evita di riscaricarlo a ogni apertura del job.
        headers={"Cache-Control": "private, max-age=86400", "Accept-Ranges": "bytes"},
    )


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
    return HTMLResponse(
        index_file.read_text(encoding="utf-8"),
        # L'interfaccia è un file solo che cambia a ogni rilascio: senza questa
        # intestazione il browser può servire la copia in cache anche dopo un
        # rebuild, e si finisce a cercare nel container un problema che sta nel
        # browser. Costa una richiesta da poche decine di kB.
        headers={"Cache-Control": "no-store, must-revalidate"},
    )
