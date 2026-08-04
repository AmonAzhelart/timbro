"""Analisi della riunione tramite LLM locale (Ollama).

Strategia:
* riunione corta (entra nel contesto) -> singola chiamata "reduce";
* riunione lunga -> map-reduce: appunti per blocco, poi sintesi finale.

L'output è vincolato via `format` (JSON Schema) supportato da Ollama, con
fallback su estrazione manuale del JSON se il modello sbaglia comunque.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

import httpx

from . import overlap, prompts
from .config import settings
from .models import MeetingAnalysis, Segment

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, str], None]

# JSON Schema passato a Ollama per forzare la struttura dell'output.
ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "overview": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
            },
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["decision"],
            },
        },
        "action_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": "string"},
                    "due": {"type": "string"},
                    "priority": {"type": "string", "enum": ["alta", "media", "bassa"]},
                },
                "required": ["task", "owner", "due", "priority"],
            },
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "overview", "sections", "decisions", "action_points"],
}


# ---------------------------------------------------------------------------
# Client Ollama
# ---------------------------------------------------------------------------
def _chat(
    user_prompt: str,
    *,
    json_schema: dict | None = None,
    temperature: float = 0.2,
    system: str | None = None,
) -> str:
    payload: dict = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system or prompts.SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,  # disattiva il reasoning verboso sui modelli ibridi (qwen3, ...)
        # Permanenza in VRAM decisa qui e non dall'ambiente del container:
        # con un LLM da 18-20 GB su una scheda da 24 è la differenza fra un
        # job successivo che parte e uno che va in out-of-memory.
        "keep_alive": settings.ollama_keep_alive,
        "options": {
            "temperature": temperature,
            "num_ctx": settings.ollama_num_ctx,
        },
    }
    if json_schema is not None:
        payload["format"] = json_schema

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    with httpx.Client(timeout=settings.ollama_timeout_s) as client:
        response = client.post(url, json=payload)

        # I modelli non "thinking" rifiutano il parametro `think`: riprova senza.
        if response.status_code == 400 and "think" in payload:
            payload.pop("think")
            response = client.post(url, json=payload)

        if response.status_code == 404:
            raise RuntimeError(
                f"Modello '{settings.ollama_model}' non trovato su Ollama. "
                f"Esegui: docker compose exec ollama ollama pull {settings.ollama_model}"
            )
        response.raise_for_status()
        data = response.json()
    return (data.get("message") or {}).get("content", "").strip()


def check_ollama() -> dict:
    """Diagnostica usata dall'endpoint /api/health."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        with httpx.Client(timeout=10) as client:
            models = [m["name"] for m in client.get(url).json().get("models", [])]
    except Exception as exc:
        return {"reachable": False, "error": str(exc), "models": []}

    base = settings.ollama_model.split(":")[0]
    ready = any(m == settings.ollama_model or m.startswith(f"{base}:") for m in models)
    return {"reachable": True, "models": models, "model_ready": ready}


# ---------------------------------------------------------------------------
# Ricognizione: deduce contesto e vocabolario dal testo grezzo
# ---------------------------------------------------------------------------
RECON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "domain": {"type": "string"},
        "languages": {"type": "array", "items": {"type": "string"}},
        # Ogni termine è ancorato al testo da cui è stato ricavato: senza
        # l'ancora non possiamo distinguere una correzione da un'invenzione.
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "heard": {"type": "string"},
                },
                "required": ["term", "heard"],
            },
        },
        "confidence": {"type": "string", "enum": ["alta", "media", "bassa"]},
    },
    "required": ["topic", "terms", "confidence"],
}


def infer_context(draft: str) -> dict:
    """Da una trascrizione grezza ricava `{topic, domain, terms, confidence}`.

    Non solleva mai: se l'LLM non è disponibile o risponde male, restituisce un
    risultato vuoto e la trascrizione prosegue con il solo contesto manuale.
    """
    empty = {"topic": "", "domain": "", "languages": [], "terms": [], "confidence": "bassa"}
    if not draft.strip():
        return empty

    try:
        raw = _chat(
            prompts.RECON.format(draft=draft),
            json_schema=RECON_SCHEMA,
            temperature=0.1,
            system=prompts.RECON_SYSTEM,
        )
    except Exception as exc:
        log.warning("Ricognizione LLM non riuscita: %s", exc)
        return {**empty, "error": str(exc)}

    data = _extract_json(raw)
    if not isinstance(data, dict):
        log.warning("Ricognizione: risposta non interpretabile")
        return empty

    # I modelli più piccoli a volte ignorano lo schema e restituiscono stringhe
    # semplici: le accettiamo trattando il termine come ancora di se stesso.
    terms: list[dict] = []
    for item in data.get("terms") or []:
        if isinstance(item, dict) and item.get("term"):
            terms.append(
                {
                    "term": str(item["term"]),
                    "heard": str(item.get("heard") or item["term"]),
                }
            )
        elif isinstance(item, str) and item.strip():
            terms.append({"term": item, "heard": item})

    return {
        "topic": str(data.get("topic") or "").strip(),
        "domain": str(data.get("domain") or "").strip(),
        "languages": [str(x) for x in (data.get("languages") or []) if x],
        "terms": terms,
        "confidence": str(data.get("confidence") or "bassa").lower(),
    }


def _extract_json(raw: str) -> Any:
    """Estrae il primo oggetto JSON bilanciato da una risposta del modello."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------------------
# Formattazione della trascrizione
# ---------------------------------------------------------------------------
def format_transcript(segments: list[Segment], speaker_names: dict[str, str] | None = None) -> str:
    """Rende la trascrizione nella forma che legge l'LLM.

    I tratti sovrapposti sono marcati: il modello deve sapere che lì
    l'attribuzione è incerta, altrimenti assegna decisioni alla persona
    sbagliata con la stessa sicurezza del resto. Le copie prodotte dalla
    duplicazione vengono però unite in una riga sola: ripetere la stessa
    frase tre volte consuma contesto e suggerisce al modello un'insistenza
    che nella riunione non c'era.
    """
    names = speaker_names or {}
    lines = []

    for group in overlap.group_for_reading(segments):
        stamp = _hhmmss(group["start"])

        if not group["overlapping"]:
            speaker = names.get(group["speakers"][0], group["speakers"][0])
            text = group["texts"][0][1] if group["texts"] else ""
            lines.append(f"[{stamp}] {speaker}: {text}")
            continue

        who = ", ".join(names.get(s, s) for s in group["speakers"])

        if group["separated"] and len(group["texts"]) > 1:
            # Tracce separate: ogni voce ha un testo suo, l'attribuzione è buona.
            for speaker, text in group["texts"]:
                display = names.get(speaker, speaker)
                lines.append(
                    f"[{stamp}] {display} [PARLANO INSIEME con {who}, "
                    f"tracce separate]: {text}"
                )
        else:
            text = group["texts"][0][1] if group["texts"] else ""
            lines.append(f"[{stamp}] [PARLATO SOVRAPPOSTO fra {who}]: {text}")

    return "\n".join(lines)


def _overlap_note(segments: list[Segment]) -> str:
    """Istruzione aggiuntiva, solo se ci sono davvero sovrapposizioni."""
    if not any(seg.overlap and not seg.separated for seg in segments):
        return ""
    return (
        "\n\nATTENZIONE: alcune righe sono marcate [PARLATO SOVRAPPOSTO fra ...]. "
        "In quei punti più persone parlavano insieme, la riga non ha un singolo "
        "parlante e il sistema non può stabilire chi abbia detto cosa. Regole "
        "per quelle righe:\n"
        "- NON attribuire a nessuno decisioni, impegni o action point che "
        "compaiono solo lì.\n"
        "- Se il contenuto è comunque rilevante, riportalo senza responsabile "
        "(owner: \"Non assegnato\") oppure inseriscilo fra le questioni aperte.\n"
        "- Non dedurre l'accordo di una persona dal solo fatto che stesse parlando."
    )


def _hhmmss(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _split_chunks(text: str, max_chars: int) -> list[str]:
    """Divide il testo su confini di riga, senza spezzare gli interventi."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        line_len = len(line) + 1
        if size + line_len > max_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Parsing robusto della risposta
# ---------------------------------------------------------------------------
def _parse_analysis(raw: str) -> MeetingAnalysis:
    data = _extract_json(raw)
    if isinstance(data, dict):
        try:
            return MeetingAnalysis.model_validate(data)
        except Exception as exc:
            log.error("JSON valido ma non conforme allo schema: %s", exc)

    log.error("Risposta LLM non parsabile: %s", raw[:500])
    return MeetingAnalysis(
        title="Riunione",
        overview=(
            "Non è stato possibile strutturare automaticamente il verbale. "
            "Testo grezzo prodotto dal modello:\n\n" + raw[:4000]
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _reduce_budget_chars() -> int:
    """Quanti caratteri di appunti possiamo passare alla fase REDUCE.

    Gli appunti della fase MAP sono volutamente dettagliati: su una riunione di
    due ore possono superare il contesto del modello. Se ciò accade Ollama
    tronca dall'inizio senza dirlo, e il verbale perde silenziosamente pezzi
    della riunione. Stimiamo il budget e, se serve, consolidiamo a gruppi.

    Circa 3 caratteri per token in italiano; lasciamo ~45% del contesto al
    prompt di sistema e alla risposta, che qui è lunga.
    """
    return max(4000, int(settings.ollama_num_ctx * 3 * 0.55))


def _fold_notes(notes: list[str], note: str, progress: ProgressFn) -> str:
    """Riduce gli appunti a un blocco che entra nel contesto, fondendo a gruppi."""
    labelled = [f"### Appunti porzione {i}\n{n}" for i, n in enumerate(notes, 1)]
    budget = _reduce_budget_chars()
    joined = "\n\n".join(labelled)
    round_no = 0

    while len(joined) > budget and len(labelled) > 1:
        round_no += 1
        progress("summarizing", 90, f"Consolidamento appunti (passata {round_no})")

        groups: list[list[str]] = []
        current: list[str] = []
        size = 0
        for block in labelled:
            # Il gruppo deve entrare nel contesto da solo: stesso budget.
            if current and size + len(block) > budget:
                groups.append(current)
                current, size = [], 0
            current.append(block)
            size += len(block) + 2
        if current:
            groups.append(current)

        # Nessun raggruppamento possibile: ogni blocco è già oltre il budget da
        # solo. Un'altra passata non ridurrebbe nulla: si esce e si tronca.
        if len(groups) == len(labelled):
            break

        labelled = [
            _chat(prompts.FOLD.format(content="\n\n".join(g)) + note, temperature=0.1)
            if len(g) > 1
            else g[0]
            for g in groups
        ]
        joined = "\n\n".join(labelled)

    # Meglio troncare qui, lasciandone traccia nei log, che farlo fare a Ollama
    # in silenzio: così almeno si sa perché il verbale è incompleto.
    if len(joined) > budget:
        log.warning(
            "Appunti troppo voluminosi per il contesto (%d caratteri, budget %d): "
            "verranno troncati. Aumenta OLLAMA_NUM_CTX o riduci CHUNK_CHARS.",
            len(joined),
            budget,
        )
        return joined[:budget]

    return joined


def analyze(
    segments: list[Segment],
    progress: ProgressFn,
    speaker_names: dict[str, str] | None = None,
) -> MeetingAnalysis:
    transcript = format_transcript(segments, speaker_names)
    note = _overlap_note(segments)
    chunks = _split_chunks(transcript, settings.chunk_chars)

    if len(chunks) == 1:
        progress("summarizing", 85, "Generazione del verbale")
        raw = _chat(
            prompts.REDUCE.format(
                source_desc="la trascrizione completa", content=transcript
            )
            + note,
            json_schema=ANALYSIS_SCHEMA,
        )
        return _parse_analysis(raw)

    # Riunione lunga: fase MAP
    notes: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        pct = 75 + int(13 * (i - 1) / len(chunks))
        progress("summarizing", pct, f"Analisi blocco {i} di {len(chunks)}")
        notes.append(
            _chat(
                prompts.MAP.format(index=i, total=len(chunks), chunk=chunk) + note,
                temperature=0.1,
            )
        )

    # Fase REDUCE
    joined = _fold_notes(notes, note, progress)
    progress("summarizing", 92, "Composizione del verbale finale")
    raw = _chat(
        prompts.REDUCE.format(
            source_desc="gli appunti estratti dalle diverse porzioni", content=joined
        )
        + note,
        json_schema=ANALYSIS_SCHEMA,
    )
    return _parse_analysis(raw)
