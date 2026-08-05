"""Task: gli impegni presi in riunione, come cose vive.

Un action point nel verbale è la registrazione di una frase detta; un task è
l'impegno che ne consegue e che qualcuno deve portare a termine. Sono due
cose diverse, e tenerle nello stesso oggetto costringeva a riscrivere il
verbale ogni volta che un impegno slittava — cioè a falsificare il registro
di ciò che era stato detto.

Qui vive la parte che non è né persistenza né API:

* **derivazione**: da quali action point nasce un task, e come non
  ricrearne di doppi quando il verbale viene rigenerato;
* **ancoraggio**: a quale istante della riunione riportare l'impegno, per
  poter riascoltare la frase in cui è stato preso.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

log = logging.getLogger(__name__)

STATI = ("da_fare", "in_corso", "bloccato", "fatto")
STATI_LABEL = {
    "da_fare": "Da fare",
    "in_corso": "In corso",
    "bloccato": "Bloccato",
    "fatto": "Fatto",
}

#: Parole troppo comuni per contribuire al riconoscimento della frase di
#: origine: sono ovunque e assomigliano a tutto.
_STOP = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da",
    "in", "con", "su", "per", "tra", "fra", "e", "o", "ma", "che", "chi",
    "cui", "non", "si", "se", "come", "dove", "quando", "piu", "meno",
    "del", "della", "dei", "delle", "dal", "dalla", "al", "alla", "ai",
    "nel", "nella", "sul", "sulla", "questo", "questa", "quello", "quella",
    "essere", "avere", "fare", "poi", "gia", "ancora", "anche", "cosa",
}


def normalize(text: str) -> str:
    lowered = (text or "").strip().lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def keywords(text: str) -> set[str]:
    """Parole significative di un testo, per confrontarlo con un altro."""
    return {
        w for w in re.findall(r"\w+", normalize(text))
        if len(w) > 3 and w not in _STOP
    }


def same_task(a: str, b: str) -> bool:
    """Due formulazioni descrivono lo stesso impegno?

    Serve alla rigenerazione del verbale: il modello riscrive le frasi con
    parole leggermente diverse, e senza questo confronto ogni rigenerazione
    creerebbe task doppi, ognuno con i propri commenti sparsi.
    """
    ka, kb = keywords(a), keywords(b)
    if not ka or not kb:
        return normalize(a) == normalize(b)
    shared = len(ka & kb)
    # Rapporto sul più piccolo dei due: "Preparare il preventivo" e
    # "Preparare il preventivo definitivo per Acme" sono lo stesso impegno.
    return shared / min(len(ka), len(kb)) >= 0.6


def find_source_time(task_text: str, segments: list[Any]) -> float | None:
    """A che punto della riunione è stato preso questo impegno.

    Il modello non restituisce istanti: si cerca il segmento che condivide più
    parole significative con la formulazione dell'impegno. È un'euristica, e
    come tale può sbagliare: per questo la soglia è alta e in mancanza di una
    corrispondenza netta si restituisce `None` invece di un punto a caso —
    un link che porta nel posto sbagliato è peggio di nessun link.
    """
    wanted = keywords(task_text)
    if not wanted or not segments:
        return None

    best_score = 0.0
    best_start: float | None = None

    for seg in segments:
        text = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")
        found = keywords(text or "")
        if not found:
            continue
        shared = len(wanted & found)
        if not shared:
            continue
        score = shared / len(wanted)
        if score > best_score:
            best_score = score
            start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", None)
            best_start = float(start) if start is not None else None

    # Sotto un terzo delle parole in comune la corrispondenza è casuale.
    return best_start if best_score >= 0.34 else None


def derive(
    analysis: Any,
    segments: list[Any],
    job_id: str,
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Quali task creare da un verbale, saltando quelli che esistono già.

    Restituisce solo i NUOVI. I task esistenti non vengono mai toccati: dopo
    una rigenerazione del verbale, commenti, stato e scadenza riprogrammata
    a mano sono lavoro umano, e sovrascriverlo con l'output di un modello
    sarebbe la cosa peggiore che questo modulo possa fare.
    """
    known = [t for t in existing if t.get("job_id") == job_id]
    fresh: list[dict[str, Any]] = []

    for index, point in enumerate(getattr(analysis, "action_points", []) or []):
        title = (getattr(point, "task", "") or "").strip()
        if not title:
            continue
        if any(same_task(title, t.get("title", "")) for t in known):
            continue

        owner = (getattr(point, "owner", "") or "").strip()
        if normalize(owner) in ("non assegnato", "nessuno", ""):
            owner = ""

        fresh.append({
            "title": title,
            "owner": owner,
            "priority": (getattr(point, "priority", "") or "media").lower(),
            "due_at": getattr(point, "due_at", "") or "",
            "due_precision": getattr(point, "due_precision", "") or "",
            "due_phrase": getattr(point, "due", "") or "",
            "job_id": job_id,
            "ap_index": index,
            "source_s": find_source_time(title, segments),
        })

    if fresh:
        log.info("Task creati dal verbale %s: %s (già presenti: %s)",
                 job_id, len(fresh), len(known))
    return fresh
