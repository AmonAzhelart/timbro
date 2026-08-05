"""Persistenza dei job su SQLite.

SQLite = zero manutenzione: nessun servizio da tenere in piedi, un solo file
dentro ./data. I payload grossi (segmenti, analisi) sono serializzati in JSON.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
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
    overlap       TEXT,
    -- Presente anche fra le MIGRATIONS: qui per i database nuovi, lì per
    -- quelli già esistenti. Ogni colonna aggiunta dopo il primo rilascio va
    -- scritta in entrambi i posti, altrimenti le due strade divergono.
    folder_id     TEXT,
    recorded_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

-- Cartelle annidate. `parent_id` NULL = radice. La profondità non è limitata
-- dallo schema ma dal codice, che rifiuta i cicli: senza quel controllo
-- bastano due spostamenti per rendere un ramo irraggiungibile.
CREATE TABLE IF NOT EXISTS folders (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    parent_id  TEXT REFERENCES folders(id) ON DELETE CASCADE,
    color      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_id);

-- Etichette trasversali: una riunione sta in una cartella sola ma può avere
-- quante etichette servono.
CREATE TABLE IF NOT EXISTS tags (
    id    TEXT PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE,
    color TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS job_tags (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_job_tags_tag ON job_tags(tag_id);

-- Glossari riusabili: argomento e vocabolario non cambiano a ogni riunione
-- dello stesso cliente, ma finora andavano riscritti a ogni caricamento.
CREATE TABLE IF NOT EXISTS glossaries (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    prompt     TEXT NOT NULL DEFAULT '',
    hotwords   TEXT NOT NULL DEFAULT '',
    folder_id  TEXT REFERENCES folders(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL
);

-- Rubrica delle voci: le stesse persone tornano, e rinominare SPEAKER_00 da
-- zero a ogni riunione è lavoro buttato.
CREATE TABLE IF NOT EXISTS known_speakers (
    name      TEXT PRIMARY KEY,
    uses      INTEGER NOT NULL DEFAULT 1,
    last_used TEXT NOT NULL
);

-- Task: gli impegni presi in riunione, come entità autonome.
--
-- Nascono dagli action point del verbale ma NON ci vivono dentro. Il verbale
-- è il registro di cosa è stato detto e non va riscritto quando riprogrammi
-- un impegno o ci aggiungi un commento; il task è la cosa viva, che si
-- sposta, si commenta e si chiude. `job_id` e `ap_index` sono il filo che
-- riporta al punto di origine, `source_s` l'istante da riascoltare.
CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    owner         TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'da_fare',
    priority      TEXT NOT NULL DEFAULT 'media',
    due_at        TEXT NOT NULL DEFAULT '',
    due_precision TEXT NOT NULL DEFAULT '',
    due_phrase    TEXT NOT NULL DEFAULT '',
    notes         TEXT NOT NULL DEFAULT '',
    job_id        TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    ap_index      INTEGER,
    source_s      REAL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    completed_at  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_at);
CREATE INDEX IF NOT EXISTS idx_tasks_job ON tasks(job_id);

-- Commenti e storico. Gli spostamenti di scadenza vengono registrati qui
-- come commenti di sistema: fra un mese, "perché era slittato" vale quanto
-- la data nuova.
CREATE TABLE IF NOT EXISTS task_comments (
    id         TEXT PRIMARY KEY,
    task_id    TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    body       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'commento',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comments_task ON task_comments(task_id);

-- Indice per la ricerca a parole esatte. `text` è l'unica colonna indicizzata:
-- le altre servono solo a riportare il risultato al suo posto.
CREATE VIRTUAL TABLE IF NOT EXISTS job_fts USING fts5(
    job_id UNINDEXED,
    kind   UNINDEXED,
    ref    UNINDEXED,
    label  UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Vettori per la ricerca per significato. Restano separati dall'indice a
-- parole: si popolano solo se il modello di embedding è disponibile, e la
-- ricerca deve continuare a funzionare quando non lo è.
CREATE TABLE IF NOT EXISTS job_vectors (
    job_id TEXT NOT NULL,
    kind   TEXT NOT NULL,
    ref    TEXT NOT NULL DEFAULT '',
    label  TEXT NOT NULL DEFAULT '',
    text   TEXT NOT NULL,
    vec    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_job ON job_vectors(job_id);
"""

#: Colonne aggiunte dopo il primo rilascio: applicate ai database già esistenti.
MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("warnings", "ALTER TABLE jobs ADD COLUMN warnings TEXT NOT NULL DEFAULT '[]'"),
    ("recon", "ALTER TABLE jobs ADD COLUMN recon TEXT"),
    ("options", "ALTER TABLE jobs ADD COLUMN options TEXT NOT NULL DEFAULT '{}'"),
    ("overlap", "ALTER TABLE jobs ADD COLUMN overlap TEXT"),
    ("folder_id", "ALTER TABLE jobs ADD COLUMN folder_id TEXT"),
    ("recorded_at", "ALTER TABLE jobs ADD COLUMN recorded_at TEXT"),
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


def _row_to_detail(row: sqlite3.Row, tags: list[str] | None = None) -> JobDetail:
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
        # Collocazione: senza queste due righe il dettaglio di una riunione
        # dichiara sempre "nessuna cartella" e nessuna etichetta, anche quando
        # l'elenco la mostra correttamente dentro la sua cartella — perché
        # l'elenco le legge e il dettaglio no.
        folder_id=row["folder_id"] if "folder_id" in row.keys() else None,
        recorded_at=(row["recorded_at"] if "recorded_at" in row.keys() else "") or "",
        tags=tags or [],
    )


def get_job(job_id: str) -> JobDetail | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        # Le etichette stanno in una tabella a parte: vanno lette qui, nella
        # stessa connessione, altrimenti il dettaglio le riporta sempre vuote.
        tags = [
            r["tag_id"]
            for r in conn.execute(
                "SELECT tag_id FROM job_tags WHERE job_id=?", (job_id,)
            ).fetchall()
        ]
    return _row_to_detail(row, tags)


def get_audio_path(job_id: str) -> str | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT audio_path FROM jobs WHERE id=?", (job_id,)).fetchone()
    return row["audio_path"] if row else None


def list_jobs(limit: int = 100) -> list[JobSummary]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT id, filename, status, progress, message, error, created_at, "
            "updated_at, duration_s, analysis, warnings, folder_id, recorded_at FROM jobs "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    tag_map = tags_by_job()
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
                folder_id=row["folder_id"],
                recorded_at=row["recorded_at"] or "",
                tags=tag_map.get(row["id"], []),
            )
        )
    return out


def delete_job(job_id: str) -> bool:
    with _lock, _connect() as conn:
        cur = conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        conn.execute("DELETE FROM job_tags WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM job_fts WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM job_vectors WHERE job_id=?", (job_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Cartelle
# ---------------------------------------------------------------------------
def list_folders() -> list[dict[str, Any]]:
    """Tutte le cartelle con il conteggio delle riunioni contenute.

    Il conteggio è quello diretto, non ricorsivo: l'albero lo compone
    l'interfaccia, che sa già quali figli sta mostrando.
    """
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT f.id, f.name, f.parent_id, f.color, f.created_at, "
            "       (SELECT COUNT(*) FROM jobs j WHERE j.folder_id = f.id) AS jobs "
            "FROM folders f ORDER BY f.name COLLATE NOCASE"
        ).fetchall()
    return [dict(r) for r in rows]


def create_folder(name: str, parent_id: str | None = None, color: str = "") -> dict[str, Any]:
    folder_id = uuid.uuid4().hex
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO folders (id, name, parent_id, color, created_at) VALUES (?,?,?,?,?)",
            (folder_id, name.strip(), parent_id or None, color, _now()),
        )
    return {"id": folder_id, "name": name.strip(), "parent_id": parent_id,
            "color": color, "created_at": _now(), "jobs": 0}


def _descendants(conn: sqlite3.Connection, folder_id: str) -> set[str]:
    """Tutti i discendenti, per impedire di spostare una cartella dentro sé stessa."""
    found: set[str] = set()
    frontier = [folder_id]
    while frontier:
        rows = conn.execute(
            f"SELECT id FROM folders WHERE parent_id IN ({','.join('?' * len(frontier))})",
            frontier,
        ).fetchall()
        frontier = [r["id"] for r in rows if r["id"] not in found]
        found.update(frontier)
    return found


def update_folder(
    folder_id: str,
    name: str | None = None,
    parent_id: str | None = None,
    color: str | None = None,
    move: bool = False,
) -> str | None:
    """Rinomina o sposta una cartella. Restituisce un errore leggibile, o None."""
    with _lock, _connect() as conn:
        if not conn.execute("SELECT 1 FROM folders WHERE id=?", (folder_id,)).fetchone():
            return "Cartella non trovata"

        if move:
            if parent_id == folder_id:
                return "Una cartella non può contenere sé stessa"
            if parent_id and parent_id in _descendants(conn, folder_id):
                return "Non puoi spostare una cartella dentro una sua sottocartella"
            if parent_id and not conn.execute(
                "SELECT 1 FROM folders WHERE id=?", (parent_id,)
            ).fetchone():
                return "Cartella di destinazione non trovata"
            conn.execute("UPDATE folders SET parent_id=? WHERE id=?", (parent_id, folder_id))

        if name is not None:
            conn.execute("UPDATE folders SET name=? WHERE id=?", (name.strip(), folder_id))
        if color is not None:
            conn.execute("UPDATE folders SET color=? WHERE id=?", (color, folder_id))
    return None


def delete_folder(folder_id: str) -> bool:
    """Elimina la cartella e le sue discendenti; le riunioni tornano alla radice.

    Cancellare riunioni insieme alla cartella sarebbe una perdita di dati
    silenziosa: si sposta il contenitore, non il contenuto.
    """
    with _lock, _connect() as conn:
        ids = {folder_id} | _descendants(conn, folder_id)
        marks = ",".join("?" * len(ids))
        conn.execute(f"UPDATE jobs SET folder_id=NULL WHERE folder_id IN ({marks})", tuple(ids))
        conn.execute(f"UPDATE glossaries SET folder_id=NULL WHERE folder_id IN ({marks})", tuple(ids))
        cur = conn.execute(f"DELETE FROM folders WHERE id IN ({marks})", tuple(ids))
        return cur.rowcount > 0


def set_job_folder(job_id: str, folder_id: str | None) -> None:
    with _lock, _connect() as conn:
        conn.execute("UPDATE jobs SET folder_id=? WHERE id=?", (folder_id or None, job_id))


# ---------------------------------------------------------------------------
# Etichette
# ---------------------------------------------------------------------------
def list_tags() -> list[dict[str, Any]]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT t.id, t.name, t.color, "
            "       (SELECT COUNT(*) FROM job_tags jt WHERE jt.tag_id = t.id) AS jobs "
            "FROM tags t ORDER BY t.name COLLATE NOCASE"
        ).fetchall()
    return [dict(r) for r in rows]


def ensure_tag(name: str, color: str = "") -> dict[str, Any]:
    """Crea l'etichetta se non esiste, altrimenti restituisce quella esistente."""
    clean = name.strip()
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT id, name, color FROM tags WHERE name=? COLLATE NOCASE", (clean,)
        ).fetchone()
        if row:
            return dict(row)
        tag_id = uuid.uuid4().hex
        conn.execute("INSERT INTO tags (id, name, color) VALUES (?,?,?)", (tag_id, clean, color))
    return {"id": tag_id, "name": clean, "color": color}


def delete_tag(tag_id: str) -> bool:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM job_tags WHERE tag_id=?", (tag_id,))
        return conn.execute("DELETE FROM tags WHERE id=?", (tag_id,)).rowcount > 0


def set_job_tags(job_id: str, tag_ids: list[str]) -> None:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM job_tags WHERE job_id=?", (job_id,))
        conn.executemany(
            "INSERT OR IGNORE INTO job_tags (job_id, tag_id) VALUES (?,?)",
            [(job_id, t) for t in tag_ids],
        )


def tags_by_job() -> dict[str, list[str]]:
    """Mappa job -> id delle sue etichette, in una sola interrogazione."""
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT job_id, tag_id FROM job_tags").fetchall()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["job_id"], []).append(r["tag_id"])
    return out


# ---------------------------------------------------------------------------
# Glossari
# ---------------------------------------------------------------------------
def list_glossaries() -> list[dict[str, Any]]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, prompt, hotwords, folder_id, created_at "
            "FROM glossaries ORDER BY name COLLATE NOCASE"
        ).fetchall()
    return [dict(r) for r in rows]


def save_glossary(
    name: str, prompt: str, hotwords: str,
    folder_id: str | None = None, glossary_id: str | None = None,
) -> dict[str, Any]:
    with _lock, _connect() as conn:
        if glossary_id:
            conn.execute(
                "UPDATE glossaries SET name=?, prompt=?, hotwords=?, folder_id=? WHERE id=?",
                (name.strip(), prompt, hotwords, folder_id or None, glossary_id),
            )
        else:
            glossary_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO glossaries (id, name, prompt, hotwords, folder_id, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (glossary_id, name.strip(), prompt, hotwords, folder_id or None, _now()),
            )
    return {"id": glossary_id, "name": name.strip(), "prompt": prompt,
            "hotwords": hotwords, "folder_id": folder_id}


def delete_glossary(glossary_id: str) -> bool:
    with _lock, _connect() as conn:
        return conn.execute("DELETE FROM glossaries WHERE id=?", (glossary_id,)).rowcount > 0


# ---------------------------------------------------------------------------
# Rubrica delle voci
# ---------------------------------------------------------------------------
def remember_speakers(names: list[str]) -> None:
    """Registra i nomi assegnati, per proporli alla riunione successiva."""
    stamp = _now()
    with _lock, _connect() as conn:
        for name in {n.strip() for n in names if n and n.strip()}:
            conn.execute(
                "INSERT INTO known_speakers (name, uses, last_used) VALUES (?,1,?) "
                "ON CONFLICT(name) DO UPDATE SET uses = uses + 1, last_used = excluded.last_used",
                (name, stamp),
            )


def known_speakers(limit: int = 60) -> list[dict[str, Any]]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT name, uses, last_used FROM known_speakers "
            "ORDER BY uses DESC, last_used DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def forget_speaker(name: str) -> bool:
    with _lock, _connect() as conn:
        return conn.execute("DELETE FROM known_speakers WHERE name=?", (name,)).rowcount > 0


# ---------------------------------------------------------------------------
# Indice di ricerca
# ---------------------------------------------------------------------------
def reindex_job(job_id: str, with_vectors: bool = True) -> dict[str, Any]:
    """Riscrive l'indice di un job. Non solleva: la ricerca non deve poter
    far fallire un'elaborazione andata a buon fine."""
    from . import search as search_mod

    job = get_job(job_id)
    if job is None:
        return {"passages": 0, "vectors": 0, "embed_error": "job non trovato"}
    try:
        with _lock, _connect() as conn:
            return search_mod.index_job(conn, job, with_vectors=with_vectors)
    except Exception as exc:
        log.warning("Indicizzazione fallita per %s: %s", job_id, exc)
        return {"passages": 0, "vectors": 0, "embed_error": str(exc)}


def reindex_all(with_vectors: bool = True) -> dict[str, Any]:
    """Ricostruisce l'indice su tutto l'archivio.

    Serve dopo un aggiornamento che introduce la ricerca su riunioni già
    esistenti, o dopo aver installato il modello di embedding.
    """
    with _lock, _connect() as conn:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM jobs WHERE status='done' ORDER BY created_at DESC"
        ).fetchall()]

    done = vectors = 0
    error: str | None = None
    for job_id in ids:
        report = reindex_job(job_id, with_vectors=with_vectors)
        done += 1
        vectors += report.get("vectors") or 0
        if report.get("embed_error") and not error:
            error = report["embed_error"]
            # Se il modello manca, insistere su cento riunioni è tempo buttato.
            with_vectors = False
    return {"jobs": done, "vectors": vectors, "embed_error": error}


def search_archive(query: str, **kwargs: Any) -> dict[str, Any]:
    from . import search as search_mod

    with _lock, _connect() as conn:
        result = search_mod.search(conn, query, **kwargs)

    # I risultati arrivano dall'indice e non conoscono la riunione: qui si
    # aggiunge ciò che serve a mostrarli (titolo, data, cartella).
    ids = {h["job_id"] for h in result["hits"]}
    if ids:
        marks = ",".join("?" * len(ids))
        with _lock, _connect() as conn:
            rows = conn.execute(
                f"SELECT id, filename, created_at, analysis, folder_id "
                f"FROM jobs WHERE id IN ({marks})", tuple(ids)
            ).fetchall()
        meta = {
            r["id"]: {
                "job_title": (json.loads(r["analysis"] or "null") or {}).get("title")
                             or r["filename"],
                "job_created": r["created_at"],
                "folder_id": r["folder_id"],
            }
            for r in rows
        }
        for hit in result["hits"]:
            hit.update(meta.get(hit["job_id"], {}))
    return result


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
def list_tasks(include_done: bool = True) -> list[dict[str, Any]]:
    """Tutti i task, con titolo della riunione di origine e numero di commenti."""
    query = (
        "SELECT t.*, "
        "  (SELECT COUNT(*) FROM task_comments c WHERE c.task_id = t.id) AS comments, "
        "  j.filename AS job_filename, j.analysis AS job_analysis "
        "FROM tasks t LEFT JOIN jobs j ON j.id = t.job_id "
    )
    if not include_done:
        query += "WHERE t.status != 'fatto' "
    query += "ORDER BY (t.due_at = '') ASC, t.due_at ASC, t.created_at ASC"

    with _lock, _connect() as conn:
        rows = conn.execute(query).fetchall()

    out = []
    for r in rows:
        item = {k: r[k] for k in r.keys() if k not in ("job_analysis", "job_filename")}
        analysis = json.loads(r["job_analysis"] or "null") or {}
        item["job_title"] = analysis.get("title") or r["job_filename"] or ""
        out.append(item)
    return out


def get_task(task_id: str) -> dict[str, Any] | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return dict(row) if row else None


def create_task(data: dict[str, Any]) -> dict[str, Any]:
    task_id = uuid.uuid4().hex
    now = _now()
    record = {
        "id": task_id,
        "title": (data.get("title") or "").strip(),
        "owner": (data.get("owner") or "").strip(),
        "status": data.get("status") or "da_fare",
        "priority": (data.get("priority") or "media").lower(),
        "due_at": data.get("due_at") or "",
        "due_precision": data.get("due_precision") or "",
        "due_phrase": data.get("due_phrase") or "",
        "notes": data.get("notes") or "",
        "job_id": data.get("job_id"),
        "ap_index": data.get("ap_index"),
        "source_s": data.get("source_s"),
        "created_at": now,
        "updated_at": now,
        "completed_at": "",
    }
    with _lock, _connect() as conn:
        conn.execute(
            f"INSERT INTO tasks ({','.join(record)}) "
            f"VALUES ({','.join('?' * len(record))})",
            tuple(record.values()),
        )
    return record


def update_task(task_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    allowed = {"title", "owner", "status", "priority", "due_at", "due_precision",
               "due_phrase", "notes"}
    clean = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not clean:
        return get_task(task_id)

    # Chiudere un task registra QUANDO: senza, "fatto la settimana scorsa" e
    # "fatto tre mesi fa" sono indistinguibili.
    if clean.get("status") == "fatto":
        clean["completed_at"] = _now()
    elif "status" in clean:
        clean["completed_at"] = ""

    clean["updated_at"] = _now()
    assignments = ", ".join(f"{k}=?" for k in clean)
    with _lock, _connect() as conn:
        cur = conn.execute(
            f"UPDATE tasks SET {assignments} WHERE id=?", (*clean.values(), task_id)
        )
        if not cur.rowcount:
            return None
    return get_task(task_id)


def delete_task(task_id: str) -> bool:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM task_comments WHERE task_id=?", (task_id,))
        return conn.execute("DELETE FROM tasks WHERE id=?", (task_id,)).rowcount > 0


def task_comments(task_id: str) -> list[dict[str, Any]]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT id, body, kind, created_at FROM task_comments "
            "WHERE task_id=? ORDER BY created_at ASC", (task_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def add_comment(task_id: str, body: str, kind: str = "commento") -> dict[str, Any]:
    comment = {
        "id": uuid.uuid4().hex,
        "task_id": task_id,
        "body": body.strip(),
        "kind": kind,
        "created_at": _now(),
    }
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO task_comments (id, task_id, body, kind, created_at) "
            "VALUES (?,?,?,?,?)",
            tuple(comment.values()),
        )
    return comment


def sync_tasks_from_job(job_id: str) -> int:
    """Crea i task mancanti a partire dagli action point di una riunione.

    Chiamata a fine elaborazione e a ogni rigenerazione del verbale. Non
    modifica né elimina i task esistenti: stato, commenti e riprogrammazioni
    sono lavoro umano, e una rigenerazione del verbale non deve cancellarlo.
    """
    from . import tasks as tasks_mod

    job = get_job(job_id)
    if job is None or job.analysis is None:
        return 0

    fresh = tasks_mod.derive(job.analysis, job.segments, job_id, list_tasks())
    for data in fresh:
        create_task(data)
    return len(fresh)


# ---------------------------------------------------------------------------
# Impegni aperti, su tutto l'archivio
# ---------------------------------------------------------------------------
def all_action_points(include_done: bool = False) -> list[dict[str, Any]]:
    """Tutti gli action point di tutte le riunioni, con la loro provenienza.

    Gli impegni vivono dentro il JSON dell'analisi: si leggono qui invece di
    duplicarli in una tabella, così restano coerenti con il verbale anche
    quando viene corretto a mano.
    """
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT id, filename, created_at, analysis FROM jobs "
            "WHERE status='done' AND analysis IS NOT NULL ORDER BY created_at DESC"
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        analysis = json.loads(row["analysis"] or "null") or {}
        title = analysis.get("title") or row["filename"]
        for i, point in enumerate(analysis.get("action_points") or []):
            if not include_done and point.get("done"):
                continue
            out.append({
                **point,
                "done": bool(point.get("done")),
                "job_id": row["id"],
                "job_title": title,
                "job_created": row["created_at"],
                "index": i,
            })
    return out


def set_action_done(job_id: str, index: int, done: bool) -> bool:
    """Spunta un impegno senza toccare il resto del verbale."""
    with _lock, _connect() as conn:
        row = conn.execute("SELECT analysis FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row or not row["analysis"]:
            return False
        analysis = json.loads(row["analysis"])
        points = analysis.get("action_points") or []
        if not 0 <= index < len(points):
            return False
        points[index]["done"] = bool(done)
        conn.execute(
            "UPDATE jobs SET analysis=?, updated_at=? WHERE id=?",
            (json.dumps(analysis, ensure_ascii=False), _now(), job_id),
        )
    return True
