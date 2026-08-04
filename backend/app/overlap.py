"""Gestione del parlato sovrapposto.

Quando due persone parlano insieme succedono due cose distinte:

1. **Whisper** produce un solo flusso di testo. È addestrato su audio a una
   voce: della sovrapposizione trascrive quella più forte e l'altra la perde,
   oppure mescola le due in una frase senza senso.
2. **WhisperX** assegna a ogni parola lo speaker con la sovrapposizione
   temporale maggiore (`assign_word_speakers`), quindi l'informazione che in
   quel punto erano in due viene semplicemente buttata via.

Qui affrontiamo il secondo problema, che è recuperabile senza modelli extra:
* spezziamo i segmenti dove cambia chi parla a livello di singola parola,
  invece di appiattire tutto sul parlante dominante;
* ricostruiamo dagli intervalli di diarizzazione dove le voci si accavallano
  davvero, e marchiamo i segmenti coinvolti.

Il primo problema si risolve solo separando le tracce audio: se ne occupa il
servizio `separator`, opzionale.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: Sotto questa durata una sovrapposizione è rumore di confine fra due turni,
#: non due persone che parlano davvero insieme.
MIN_OVERLAP_S = 0.4


# ---------------------------------------------------------------------------
# 1. Segmenti spezzati sul cambio di voce
# ---------------------------------------------------------------------------
def split_on_speaker_change(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Divide i segmenti dove le parole cambiano parlante.

    WhisperX etichetta ogni parola, ma il segmento riceve un solo speaker: il
    dominante. In uno scambio serrato ("— hai finito? — sì, ieri") l'intero
    scambio finisce attribuito a una sola persona. Qui usiamo le etichette di
    parola, che sono già disponibili e più precise.

    I segmenti senza parole etichettate restano invariati.
    """
    out: list[dict[str, Any]] = []

    for seg in segments:
        words = [w for w in (seg.get("words") or []) if (w.get("word") or "").strip()]
        labelled = [w for w in words if w.get("speaker")]

        # Senza etichette di parola non c'è nulla da spezzare
        if not words or len(labelled) < 2:
            out.append(seg)
            continue

        # Le parole senza etichetta ereditano dalla precedente: sono in genere
        # punteggiatura o parole brevi in mezzo a un turno.
        current = labelled[0]["speaker"]
        for w in words:
            if w.get("speaker"):
                current = w["speaker"]
            else:
                w["speaker"] = current

        groups: list[list[dict]] = []
        for w in words:
            if groups and groups[-1][0]["speaker"] == w["speaker"]:
                groups[-1].append(w)
            else:
                groups.append([w])

        if len(groups) == 1:
            seg["speaker"] = groups[0][0]["speaker"]
            out.append(seg)
            continue

        for group in groups:
            text = " ".join((w.get("word") or "").strip() for w in group).strip()
            if not text:
                continue
            starts = [w["start"] for w in group if w.get("start") is not None]
            ends = [w["end"] for w in group if w.get("end") is not None]
            out.append(
                {
                    "start": min(starts) if starts else seg.get("start", 0.0),
                    "end": max(ends) if ends else seg.get("end", 0.0),
                    "speaker": group[0]["speaker"],
                    "text": text,
                    "words": group,
                }
            )

    if len(out) != len(segments):
        log.info("Segmenti spezzati sul cambio di voce: %s -> %s", len(segments), len(out))
    return out


# ---------------------------------------------------------------------------
# 2. Intervalli in cui parlano in più di uno
# ---------------------------------------------------------------------------
def find_overlaps(diarize_df: Any) -> list[dict[str, Any]]:
    """Dagli intervalli di diarizzazione ricava dove le voci si accavallano.

    `diarize_df` è il DataFrame di WhisperX con colonne start/end/speaker.
    Restituisce intervalli `{start, end, speakers}` con almeno due voci attive.
    """
    try:
        rows = [
            (float(r["start"]), float(r["end"]), str(r["speaker"]))
            for _, r in diarize_df.iterrows()
        ]
    except Exception as exc:
        log.warning("Intervalli di diarizzazione non leggibili: %s", exc)
        return []

    if len(rows) < 2:
        return []

    # Sweep line sui confini: fra due confini consecutivi l'insieme delle voci
    # attive è costante, quindi basta contarle una volta per intervallo.
    boundaries = sorted({t for start, end, _ in rows for t in (start, end)})
    overlaps: list[dict[str, Any]] = []

    for a, b in zip(boundaries, boundaries[1:]):
        if b - a <= 0:
            continue
        active = sorted({spk for start, end, spk in rows if start < b and end > a})
        if len(active) < 2:
            continue
        prev = overlaps[-1] if overlaps else None
        # Intervalli contigui con le stesse voci vengono fusi
        if prev and prev["speakers"] == active and abs(prev["end"] - a) < 1e-6:
            prev["end"] = b
        else:
            overlaps.append({"start": a, "end": b, "speakers": active})

    significant = [o for o in overlaps if o["end"] - o["start"] >= MIN_OVERLAP_S]
    if significant:
        total = sum(o["end"] - o["start"] for o in significant)
        log.info(
            "Sovrapposizioni: %s intervalli, %.0fs complessivi", len(significant), total
        )
    return significant


def summarize(overlaps: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    """Statistiche mostrate nell'interfaccia."""
    total = sum(o["end"] - o["start"] for o in overlaps)
    max_speakers = max((len(o["speakers"]) for o in overlaps), default=0)
    return {
        "count": len(overlaps),
        "total_s": round(total, 1),
        "percent": round(total * 100 / duration_s, 1) if duration_s else 0.0,
        "max_speakers": max_speakers,
    }


# ---------------------------------------------------------------------------
# 3. Marcatura e duplicazione dei segmenti sovrapposti
# ---------------------------------------------------------------------------
def _speakers_at(overlaps: list[dict[str, Any]], start: float, end: float) -> list[str]:
    """Voci attive durante un segmento, se ricade in una zona di sovrapposizione."""
    found: set[str] = set()
    for o in overlaps:
        if o["start"] < end and o["end"] > start:
            # Solo se la sovrapposizione copre una parte non trascurabile
            shared = min(o["end"], end) - max(o["start"], start)
            if shared >= min(MIN_OVERLAP_S, (end - start) * 0.5):
                found.update(o["speakers"])
    return sorted(found)


def group_for_reading(segments: list[Any]) -> list[dict[str, Any]]:
    """Raggruppa i segmenti in eventi leggibili, ricucendo le duplicazioni.

    `annotate(duplicate=True)` emette lo stesso testo una volta per voce
    presente. È corretto come dato — dice chi c'era — ma illeggibile come
    testo: la stessa frase compare tre volte di fila attribuita a tre persone
    diverse, e chi legge pensa che la trascrizione sia rotta.

    Qui quelle copie tornano a essere un evento solo. Ogni gruppo è
    ``{"start", "end", "speakers", "texts", "overlapping", "separated"}``:

    * ``overlapping=False`` → intervento normale, una voce e un testo;
    * ``overlapping=True`` con un solo testo → più voci insieme, tracce non
      separate: il testo è l'unico flusso che Whisper ha distinto e non va
      attribuito a nessuno in particolare;
    * ``overlapping=True`` con più testi → tracce separate, ogni voce ha
      davvero detto una cosa diversa.

    Accetta sia i `Segment` del modello sia i dizionari grezzi, perché serve
    sia all'esportazione sia alla costruzione del prompt.
    """
    def get(seg: Any, key: str, default: Any) -> Any:
        value = seg.get(key, default) if isinstance(seg, dict) else getattr(seg, key, default)
        return default if value is None else value

    groups: list[dict[str, Any]] = []
    index = 0
    while index < len(segments):
        seg = segments[index]
        active = get(seg, "overlap", [])

        if len(active) < 2:
            groups.append(
                {
                    "start": get(seg, "start", 0.0),
                    "end": get(seg, "end", 0.0),
                    "speakers": [get(seg, "speaker", "?")],
                    "texts": [(get(seg, "speaker", "?"), get(seg, "text", ""))],
                    "overlapping": False,
                    "separated": False,
                }
            )
            index += 1
            continue

        # Consecutivi e temporalmente intrecciati: sono lo stesso evento.
        members = [seg]
        end = get(seg, "end", 0.0)
        index += 1
        while index < len(segments):
            nxt = segments[index]
            if len(get(nxt, "overlap", [])) < 2 or get(nxt, "start", 0.0) >= end:
                break
            end = max(end, get(nxt, "end", 0.0))
            members.append(nxt)
            index += 1

        # Il dominante per primo: è la voce di cui il testo è più attendibile.
        speakers: list[str] = []
        primary = None
        for member in members:
            who = get(member, "speaker", "?")
            if who not in speakers:
                speakers.append(who)
            if primary is None and get(member, "overlap_primary", True):
                primary = who
        if primary and primary in speakers:
            speakers = [primary] + [s for s in speakers if s != primary]

        seen: set[str] = set()
        texts: list[tuple[str, str]] = []
        for member in members:
            text = (get(member, "text", "") or "").strip()
            if text and text not in seen:
                seen.add(text)
                texts.append((get(member, "speaker", "?"), text))

        groups.append(
            {
                "start": get(members[0], "start", 0.0),
                "end": end,
                "speakers": speakers,
                "texts": texts,
                "overlapping": True,
                "separated": any(get(m, "separated", False) for m in members),
            }
        )

    return groups


def annotate(
    segments: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    duplicate: bool = True,
) -> list[dict[str, Any]]:
    """Marca i segmenti sovrapposti e, se richiesto, ne emette uno per voce.

    Con `duplicate=True` un tratto in cui parlano A e B produce due righe con
    lo stesso testo, una per ciascuno: si sa chi c'era, ma il testo resta
    quello che Whisper è riuscito a sentire. La resa completa — testo diverso
    per ciascuna voce — richiede la separazione delle tracce audio.
    """
    if not overlaps:
        return segments

    out: list[dict[str, Any]] = []
    for seg in segments:
        active = _speakers_at(overlaps, seg.get("start", 0.0), seg.get("end", 0.0))
        if len(active) < 2:
            out.append(seg)
            continue

        seg["overlap"] = active
        if not duplicate:
            out.append(seg)
            continue

        # Il parlante dominante resta primo: è quello di cui il testo è più
        # probabilmente fedele.
        dominant = seg.get("speaker")
        ordered = ([dominant] if dominant in active else []) + [
            s for s in active if s != dominant
        ]
        for i, speaker in enumerate(ordered):
            copy = dict(seg)
            copy["speaker"] = speaker
            copy["overlap"] = active
            copy["overlap_primary"] = i == 0
            out.append(copy)

    return out
