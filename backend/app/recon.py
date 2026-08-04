"""Ricognizione: capire di cosa si parla prima di trascrivere sul serio.

Whisper decodifica ogni spezzone senza contesto, quindi sbaglia proprio dove
conta (nomi, sigle, termini stranieri). Qui facciamo un giro di ricognizione
economico — pochi minuti di audio campionati, modello leggero — e passiamo il
testo grezzo a un LLM che ne deduce argomento e vocabolario. Il risultato
diventa l'`initial_prompt` della trascrizione vera.

Il testo di ricognizione è volutamente scadente: serve a capire il DOMINIO,
non a essere letto. Per questo l'estrazione dei termini è conservativa, con
divieto esplicito di inventare e un filtro sulle parole troppo storpiate.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

import numpy as np

from .config import settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

#: Modello usato per la ricognizione: veloce, la qualità qui conta poco.
RECON_MODEL = "small"

#: Durata di ogni spezzone campionato, in secondi.
WINDOW_S = 120

#: Numero massimo di spezzoni. Con 5 finestre da 2 minuti si coprono 10 minuti
#: distribuiti sull'intera riunione: abbastanza per inquadrare il tema.
MAX_WINDOWS = 5


def sample_windows(audio: np.ndarray, duration_s: float) -> np.ndarray:
    """Estrae spezzoni distribuiti uniformemente e li concatena.

    Saltiamo i primi 15 secondi: di solito sono convenevoli e rumore di setup,
    poco rappresentativi del contenuto.
    """
    skip = min(15.0, duration_s * 0.05)
    usable = duration_s - skip
    if usable <= 0:
        return audio

    # Riunione breve: tanto vale usarla tutta.
    if usable <= WINDOW_S * 2:
        return audio[int(skip * SAMPLE_RATE) :]

    windows = min(MAX_WINDOWS, max(2, int(usable // WINDOW_S)))
    step = usable / windows
    chunks: list[np.ndarray] = []
    for i in range(windows):
        start = skip + i * step
        a = int(start * SAMPLE_RATE)
        b = int(min(start + WINDOW_S, duration_s) * SAMPLE_RATE)
        if b > a:
            chunks.append(audio[a:b])

    sampled = np.concatenate(chunks) if chunks else audio
    log.info(
        "Ricognizione: %s finestre da %ss su %.0fs totali (%.0fs campionati)",
        len(chunks), WINDOW_S, duration_s, len(sampled) / SAMPLE_RATE,
    )
    return sampled


# ---------------------------------------------------------------------------
# Pulizia del testo grezzo
# ---------------------------------------------------------------------------
def clean_draft(segments: list[dict[str, Any]], max_chars: int = 8000) -> str:
    """Concatena i segmenti della ricognizione in un testo compatto."""
    parts: list[str] = []
    total = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return " ".join(parts)


#: Parole comuni che non hanno valore come hotword.
_STOPWORDS = {
    "questo", "quello", "allora", "perché", "perche", "quindi", "adesso",
    "comunque", "praticamente", "sostanzialmente", "diciamo", "insomma",
    "niente", "cioè", "cioe", "ecco", "appunto", "magari", "proprio",
}


#: Somiglianza minima fra il termine corretto e la forma effettivamente sentita.
#: Volutamente permissiva: il controllo che fa il lavoro è l'ancoraggio al testo,
#: questo serve solo a impedire accostamenti assurdi ("il" -> "Kubernetes").
MIN_SIMILARITY = 0.45


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "").lower())


def filter_terms(terms: list[dict], draft: str) -> list[dict]:
    """Tiene solo i termini ancorati a testo realmente presente nella ricognizione.

    Un hotword sbagliato è peggio di nessun hotword: Whisper lo privilegia,
    quindi trasforma un errore occasionale in un errore sistematico.

    Il criterio NON è la somiglianza col testo. Misurarla non discrimina: su un
    caso reale "environment" (recupero corretto da "envenot") e "Salesforce"
    (invenzione che assomiglia a "salvatore") si collocano a quattro centesimi
    di distanza. Qualunque soglia che tenga il primo lascia passare il secondo.

    Il criterio è invece che l'LLM dichiari da QUALE punto del testo ricava il
    termine, e che quel punto esista davvero. Questo elimina l'errore più
    frequente — aggiungere tecnologie plausibili mai nominate — perché
    costringerebbe il modello a citare testo inesistente.

    Resta possibile un abbinamento sbagliato fra ancora reale e termine errato.
    Per questo restituiamo anche l'ancora: mostrandola in revisione, un
    "Salesforce ← salvatore" salta all'occhio e si toglie in un clic.
    """
    kept: list[dict] = []
    seen: set[str] = set()
    normalized_draft = _normalize(draft)

    for item in terms:
        if isinstance(item, str):
            item = {"term": item, "heard": item}
        if not isinstance(item, dict):
            continue

        term = re.sub(r"\s+", " ", str(item.get("term") or "")).strip(" .,;:!?\"'()[]{}")
        heard = re.sub(r"\s+", " ", str(item.get("heard") or "")).strip()
        if not term:
            continue

        key = term.lower()
        if key in seen or key in _STOPWORDS:
            continue
        if len(term) < 3 or len(term) > 40:
            continue
        # Solo lettere, cifre, spazi e i separatori tipici dei nomi propri
        if not re.fullmatch(r"[\w0-9 .\-'/&+]+", term, flags=re.UNICODE):
            continue
        # Un hotword è una parola o due, non una frase
        if len(term.split()) > 3:
            continue

        # (a) L'ancora deve esistere davvero negli estratti trascritti.
        anchor = _normalize(heard).strip()
        if not anchor or anchor not in normalized_draft:
            log.debug("Termine scartato, ancora inesistente: %r <- %r", term, heard)
            continue

        # (b) Termine e ancora devono essere plausibilmente la stessa cosa.
        if difflib.SequenceMatcher(None, key, anchor).ratio() < MIN_SIMILARITY:
            log.debug("Termine scartato, ancora incoerente: %r <- %r", term, heard)
            continue

        kept.append({"term": term, "heard": heard, "corrected": key != anchor.strip()})
        seen.add(key)

    return kept[:40]


def terms_text(terms: list[dict]) -> str:
    """Dai termini filtrati alla stringa di hotwords per Whisper."""
    return ", ".join(t["term"] for t in terms)


def merge_context(auto: str, manual: str) -> str:
    """Il contesto scritto a mano ha la precedenza, quello dedotto lo integra."""
    auto, manual = (auto or "").strip(), (manual or "").strip()
    if not auto:
        return manual
    if not manual:
        return auto
    if auto.lower() in manual.lower():
        return manual
    return f"{manual} {auto}"


def merge_terms(auto: list[dict] | list[str], manual: str) -> str:
    """Unisce i termini dedotti a quelli inseriti dall'utente, senza duplicati."""
    auto_names = [t["term"] if isinstance(t, dict) else str(t) for t in auto]
    out: list[str] = []
    seen: set[str] = set()
    for term in [t.strip() for t in (manual or "").split(",")] + auto_names:
        term = term.strip()
        if term and term.lower() not in seen:
            out.append(term)
            seen.add(term.lower())
    return ", ".join(out)
