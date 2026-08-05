"""Ricerca su tutto l'archivio: parole esatte e significato.

Due meccanismi, perché servono a due domande diverse.

* **Parole esatte** (SQLite FTS5). "Salvatore Mazzaglia", "flipper zero", una
  cifra, una sigla. Immediato, non richiede modelli, funziona sempre. È il
  modo giusto per ritrovare qualcosa che *sai* essere stato detto.
* **Significato** (embedding via Ollama). "quella volta che si discuteva se
  comprare l'hardware invece di noleggiarlo": nessuna di queste parole compare
  necessariamente nella trascrizione. Richiede un modello di embedding.

I due insiemi vengono uniti e riordinati. Se il modello non è installato la
ricerca continua a funzionare con le sole parole esatte, dicendolo: una
funzione che sparisce senza spiegazioni è peggio di una funzione assente.

L'indice si popola a fine elaborazione e a ogni modifica del verbale. Non
esiste un passaggio di ricostruzione automatico all'avvio perché su archivi
grandi sarebbe una sorpresa costosa: c'è una rotta esplicita.
"""

from __future__ import annotations

import json
import logging
import math
import re
import struct
from typing import Any

import httpx

from .config import settings
from .models import JobDetail

log = logging.getLogger(__name__)

#: Durata di un blocco di trascrizione nell'indice. Troppo corto e il
#: frammento non ha senso da solo; troppo lungo e non si capisce a che punto
#: della riunione riportarti.
CHUNK_SECONDS = 45.0

#: Etichette leggibili per la provenienza di un risultato.
KIND_LABELS = {
    "titolo": "Titolo",
    "sintesi": "Sintesi",
    "argomento": "Argomento",
    "decisione": "Decisione",
    "impegno": "Action point",
    "questione": "Questione aperta",
    "trascrizione": "Trascrizione",
}


# ---------------------------------------------------------------------------
# Estrazione dei brani indicizzabili da un job
# ---------------------------------------------------------------------------
def extract_passages(job: JobDetail) -> list[dict[str, Any]]:
    """Scompone un job nei brani che ha senso cercare separatamente.

    Ogni brano porta con sé l'istante a cui riportare l'utente: un risultato
    di ricerca che non ti dice *dove* riascoltare vale metà.
    """
    out: list[dict[str, Any]] = []
    analysis = job.analysis

    if analysis:
        if analysis.title:
            out.append({"kind": "titolo", "ref": "", "label": analysis.title,
                        "text": analysis.title})
        if analysis.overview:
            out.append({"kind": "sintesi", "ref": "", "label": "Sintesi",
                        "text": analysis.overview})
        for i, section in enumerate(analysis.sections):
            out.append({
                "kind": "argomento", "ref": str(i), "label": section.title,
                "text": f"{section.title}\n{section.content}",
            })
        for i, decision in enumerate(analysis.decisions):
            out.append({
                "kind": "decisione", "ref": str(i), "label": decision.decision,
                "text": f"{decision.decision}\n{decision.context}".strip(),
            })
        for i, point in enumerate(analysis.action_points):
            out.append({
                "kind": "impegno", "ref": str(i), "label": point.task,
                "text": f"{point.task} — {point.owner} — {point.due}",
            })
        for i, question in enumerate(analysis.open_questions):
            out.append({"kind": "questione", "ref": str(i), "label": question,
                        "text": question})

    # Trascrizione a blocchi di durata fissa, con il nome di chi parla: senza,
    # il frammento è un muro di testo senza attribuzione.
    names = job.speaker_names or {}
    chunk: list[str] = []
    chunk_start: float | None = None
    last_end = 0.0

    def flush() -> None:
        if chunk and chunk_start is not None:
            out.append({
                "kind": "trascrizione", "ref": f"{chunk_start:.2f}",
                "label": _stamp(chunk_start), "text": " ".join(chunk),
            })

    for seg in job.segments:
        if chunk_start is None:
            chunk_start = seg.start
        if seg.start - chunk_start >= CHUNK_SECONDS:
            flush()
            chunk, chunk_start = [], seg.start
        who = names.get(seg.speaker, seg.speaker)
        chunk.append(f"{who}: {seg.text}")
        last_end = seg.end
    flush()

    return [p for p in out if (p.get("text") or "").strip()]


def _stamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Indicizzazione
# ---------------------------------------------------------------------------
def index_job(conn, job: JobDetail, with_vectors: bool = True) -> dict[str, Any]:
    """(Ri)costruisce l'indice per un singolo job.

    `conn` arriva dal chiamante perché l'indicizzazione partecipa alla stessa
    transazione del salvataggio: un job salvato ma non indicizzato sarebbe
    invisibile alla ricerca fino al riavvio.
    """
    passages = extract_passages(job)

    conn.execute("DELETE FROM job_fts WHERE job_id=?", (job.id,))
    conn.executemany(
        "INSERT INTO job_fts (job_id, kind, ref, label, text) VALUES (?,?,?,?,?)",
        [(job.id, p["kind"], p["ref"], p["label"], p["text"]) for p in passages],
    )

    report = {"passages": len(passages), "vectors": 0, "embed_error": None}
    if not with_vectors or not settings.semantic_search:
        return report

    try:
        vectors = embed_many([p["text"] for p in passages])
    except Exception as exc:
        log.warning("Indice semantico non aggiornato per %s: %s", job.id, exc)
        report["embed_error"] = str(exc)
        return report

    conn.execute("DELETE FROM job_vectors WHERE job_id=?", (job.id,))
    conn.executemany(
        "INSERT INTO job_vectors (job_id, kind, ref, label, text, vec) VALUES (?,?,?,?,?,?)",
        [
            (job.id, p["kind"], p["ref"], p["label"], p["text"], pack(v))
            for p, v in zip(passages, vectors)
            if v
        ],
    )
    report["vectors"] = len(vectors)
    return report


# ---------------------------------------------------------------------------
# Embedding tramite Ollama
# ---------------------------------------------------------------------------
def embed_many(texts: list[str]) -> list[list[float]]:
    """Vettorializza più testi in una sola richiesta.

    Ollama espone `/api/embed` (plurale) dalla 0.3.4; le versioni precedenti
    avevano `/api/embeddings`, un testo per volta. Si prova la prima e si
    ripiega sulla seconda, invece di pretendere una versione minima.
    """
    if not texts:
        return []

    base = settings.ollama_base_url.rstrip("/")
    model = settings.embed_model

    with httpx.Client(timeout=settings.ollama_timeout_s) as client:
        response = client.post(f"{base}/api/embed", json={"model": model, "input": texts})
        if response.status_code == 404:
            out = []
            for text in texts:
                r = client.post(
                    f"{base}/api/embeddings", json={"model": model, "prompt": text}
                )
                r.raise_for_status()
                out.append(r.json().get("embedding") or [])
            return out
        if response.status_code == 404 or response.status_code == 400:
            raise RuntimeError(
                f"Modello di embedding '{model}' non disponibile su Ollama. "
                f"Scaricalo con: docker compose exec ollama ollama pull {model}"
            )
        response.raise_for_status()
        return response.json().get("embeddings") or []


def pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Interrogazione
# ---------------------------------------------------------------------------
def _fts_query(raw: str) -> str:
    """Traduce quello che scrive una persona in una query FTS5 valida.

    Gli operatori di FTS5 (virgolette, asterischi, NEAR, due punti) fanno
    fallire la query se arrivano per caso: qui si tengono solo le parole e si
    aggiunge il prefisso, perché chi cerca "preventiv" si aspetta "preventivo".
    """
    words = re.findall(r"\w+", raw, flags=re.UNICODE)
    if not words:
        return ""
    return " ".join(f'"{w}"*' for w in words if len(w) > 1)


def _snippet(text: str, needles: list[str], width: int = 220) -> str:
    """Ritaglia il testo attorno alla prima parola trovata."""
    low = text.lower()
    pos = min(
        (low.find(n) for n in needles if n and low.find(n) >= 0),
        default=-1,
    )
    if pos < 0:
        return text[:width] + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def search(
    conn,
    query: str,
    limit: int = 40,
    kinds: list[str] | None = None,
    folder_ids: list[str] | None = None,
    job_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Cerca su tutto l'archivio unendo parole esatte e significato.

    Il punteggio combina i due segnali invece di sceglierne uno: una
    corrispondenza esatta è quasi sempre ciò che si voleva, ma la vicinanza di
    significato ripesca ciò che è stato detto con altre parole.
    """
    query = (query or "").strip()
    if not query:
        return {"hits": [], "semantic": False, "note": None}

    allowed: set[str] | None = None
    if job_ids is not None:
        allowed = set(job_ids)

    hits: dict[tuple[str, str, str], dict[str, Any]] = {}
    needles = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 1]

    # --- 1. Parole esatte ---
    expression = _fts_query(query)
    if expression:
        try:
            rows = conn.execute(
                "SELECT job_id, kind, ref, label, text, rank FROM job_fts "
                "WHERE job_fts MATCH ? ORDER BY rank LIMIT ?",
                (expression, limit * 4),
            ).fetchall()
        except Exception as exc:
            log.warning("Ricerca a parole non riuscita: %s", exc)
            rows = []

        for r in rows:
            if allowed is not None and r["job_id"] not in allowed:
                continue
            if kinds and r["kind"] not in kinds:
                continue
            key = (r["job_id"], r["kind"], r["ref"])
            # `rank` di FTS5 è negativo e più basso = più pertinente.
            hits[key] = {
                "job_id": r["job_id"], "kind": r["kind"], "ref": r["ref"],
                "label": r["label"], "text": r["text"],
                "exact": min(1.0, abs(float(r["rank"] or 0)) / 10.0),
                "semantic": 0.0,
            }

    # --- 2. Significato ---
    semantic_ok = False
    note = None
    if settings.semantic_search:
        try:
            wanted = embed_many([query])[0]
            rows = conn.execute(
                "SELECT job_id, kind, ref, label, text, vec FROM job_vectors"
            ).fetchall()
            if rows:
                semantic_ok = True
                scored = []
                for r in rows:
                    if allowed is not None and r["job_id"] not in allowed:
                        continue
                    if kinds and r["kind"] not in kinds:
                        continue
                    scored.append((cosine(wanted, unpack(r["vec"])), r))
                scored.sort(key=lambda x: x[0], reverse=True)
                for score, r in scored[:limit * 2]:
                    # Sotto questa soglia la somiglianza è rumore: due testi
                    # qualsiasi in italiano si somigliano un po'.
                    if score < 0.45:
                        break
                    key = (r["job_id"], r["kind"], r["ref"])
                    if key in hits:
                        hits[key]["semantic"] = score
                    else:
                        hits[key] = {
                            "job_id": r["job_id"], "kind": r["kind"], "ref": r["ref"],
                            "label": r["label"], "text": r["text"],
                            "exact": 0.0, "semantic": score,
                        }
            else:
                note = ("Indice per significato vuoto: usa «Ricostruisci indice» "
                        "nelle impostazioni per abilitarlo sulle riunioni esistenti.")
        except Exception as exc:
            note = f"Ricerca per significato non disponibile: {exc}"
            log.info("Ricerca semantica saltata: %s", exc)

    # --- 3. Unione ---
    results = list(hits.values())
    for h in results:
        # La corrispondenza esatta pesa più della somiglianza: quando c'è, è
        # quasi certamente ciò che si cercava.
        h["score"] = round(h["exact"] * 1.0 + h["semantic"] * 0.75, 4)
        h["snippet"] = _snippet(h["text"], needles)
        h["kind_label"] = KIND_LABELS.get(h["kind"], h["kind"])
        h.pop("text", None)
    results.sort(key=lambda h: h["score"], reverse=True)

    return {"hits": results[:limit], "semantic": semantic_ok, "note": note}
