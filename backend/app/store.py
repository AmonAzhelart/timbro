"""Persistenza dei job su SQLite.

SQLite = zero manutenzione: nessun servizio da tenere in piedi, un solo file
dentro ./data. I payload grossi (segmenti, analisi) sono serializzati in JSON.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from .config import settings
from .models import (
    JobDetail,
    JobOptions,
    JobStatus,
    JobSummary,
    MeetingAnalysis,
    OverlapStats,
    ReconReport,
    STATUS_LABELS,
    Segment,
)

log = logging.getLogger(__name__)
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    audio_path    TEXT,
    status        TEXT NOT NULL,
    progress      INTEGER NOT NULL DEFAULT 0,
    message       TEXT NOT NULL DEFAULT '',
    error         TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    duration_s    REAL,
    language      TEXT,
    segments      TEXT,
    analysis      TEXT,
    speaker_names TEXT NOT NULL DEFAULT '{}',
    warnings      TEXT NOT NULL DEFAULT '[]',
    recon         TEXT,
    options       TEXT NOT NULL DEFAULT '{}',
    overlap       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""

#: Colonne aggiunte dopo il primo rilascio: applicate ai database già esistenti.
MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("warnings", "ALTER TABLE jobs ADD COLUMN warnings TEXT NOT NULL DEFAULT '[]'"),
    ("recon", "ALTER TABLE jobs ADD COLUMN recon TEXT"),
    ("options", "ALTER TABLE jobs ADD COLUMN options TEXT NOT NULL DEFAULT '{}'"),
    ("overlap", "ALTER TABLE jobs ADD COLUMN overlap TEXT"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    settings.ensure_dirs()
    with _lock, _connect() as conn:
        conn.executescript(SCHEMA)
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        for column, ddl in MIGRATIONS:
            if column not in existing:
                conn.execute(ddl)
                log.info("Schema aggiornato: aggiunta colonna '%s'", column)


def recover_interrupted() -> int:
    """Job rimasti appesi per un riavvio del container -> marcati in errore."""
    running = (
        JobStatus.TRANSCRIBING,
        JobStatus.ALIGNING,
        JobStatus.DIARIZING,
        JobStatus.SUMMARIZING,
        JobStatus.QUEUED,
    )
    placeholders = ",".join("?" * len(running))
    with _lock, _connect() as conn:
        cur = conn.execute(
            f"UPDATE jobs SET status=?, error=?, updated_at=? "
            f"WHERE status IN ({placeholders})",
            (
                JobStatus.ERROR.value,
                "Elaborazione interrotta dal riavvio del servizio. Rilancia il job.",
                _now(),
                *[s.value for s in running],
            ),
        )
        return cur.rowcount


def create_job(
    job_id: str, filename: str, audio_path: str, options: JobOptions | None = None
) -> None:
    now = _now()
    payload = (options or JobOptions()).model_dump_json()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, filename, audio_path, status, progress, message, "
            "created_at, updated_at, options) VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, filename, audio_path, JobStatus.QUEUED.value, 0, "In coda",
             now, now, payload),
        )


def get_options(job_id: str) -> JobOptions:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT options FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row or not row["options"]:
        return JobOptions()
    try:
        return JobOptions.model_validate_json(row["options"])
    except Exception:
        return JobOptions()


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    for key in ("segments", "analysis", "speaker_names", "warnings",
                "recon", "options", "overlap"):
        if key in fields and not isinstance(fields[key], (str, type(None))):
            fields[key] = json.dumps(fields[key], ensure_ascii=False)
    fields["updated_at"] = _now()
    assignments = ", ".join(f"{k}=?" for k in fields)
    with _lock, _connect() as conn:
        conn.execute(f"UPDATE jobs SET {assignments} WHERE id=?", (*fields.values(), job_id))


def _row_to_detail(row: sqlite3.Row) -> JobDetail:
    segments = [Segment(**s) for s in json.loads(row["segments"] or "[]")]
    analysis_raw = json.loads(row["analysis"] or "null")
    speakers = sorted({s.speaker for s in segments})
    recon_raw = json.loads(row["recon"] or "null")
    options_raw = json.loads(row["options"] or "{}")
    overlap_raw = json.loads(row["overlap"] or "null")
    return JobDetail(
        warnings=json.loads(row["warnings"] or "[]"),
        recon=ReconReport.model_validate(recon_raw) if recon_raw else None,
        options=JobOptions.model_validate(options_raw),
        overlap=OverlapStats.model_validate(overlap_raw) if overlap_raw else None,
        id=row["id"],
        filename=row["filename"],
        status=JobStatus(row["status"]),
        status_label=STATUS_LABELS.get(row["status"], row["status"]),
        progress=row["progress"],
        message=row["message"] or "",
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        duration_s=row["duration_s"],
        speakers=speakers,
        title=(analysis_raw or {}).get("title"),
        segments=segments,
        analysis=MeetingAnalysis.model_validate(analysis_raw) if analysis_raw else None,
        speaker_names=json.loads(row["speaker_names"] or "{}"),
    )


def get_job(job_id: str) -> JobDetail | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row_to_detail(row) if row else None


def get_audio_path(job_id: str) -> str | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT audio_path FROM jobs WHERE id=?", (job_id,)).fetchone()
    return row["audio_path"] if row else None


def list_jobs(limit: int = 100) -> list[JobSummary]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT id, filename, status, progress, message, error, created_at, "
            "updated_at, duration_s, analysis, warnings FROM jobs "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out: list[JobSummary] = []
    for row in rows:
        analysis = json.loads(row["analysis"] or "null")
        out.append(
            JobSummary(
                warnings=json.loads(row["warnings"] or "[]"),
                id=row["id"],
                filename=row["filename"],
                status=JobStatus(row["status"]),
                status_label=STATUS_LABELS.get(row["status"], row["status"]),
                progress=row["progress"],
                message=row["message"] or "",
                error=row["error"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                duration_s=row["duration_s"],
                title=(analysis or {}).get("title"),
            )
        )
    return out


def delete_job(job_id: str) -> bool:
    with _lock, _connect() as conn:
        cur = conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return cur.rowcount > 0
