"""Esportazione del verbale in Markdown."""

from __future__ import annotations

from . import overlap
from .models import JobDetail
from .summarize import _hhmmss

_PRIORITY_MARK = {"alta": "🔴", "media": "🟡", "bassa": "🟢"}


def to_markdown(job: JobDetail, include_transcript: bool = True) -> str:
    a = job.analysis
    names = job.speaker_names or {}
    lines: list[str] = []

    title = (a.title if a and a.title else None) or job.filename
    lines.append(f"# {title}")
    lines.append("")

    meta = [f"**File:** {job.filename}", f"**Data elaborazione:** {job.created_at[:10]}"]
    if job.duration_s:
        meta.append(f"**Durata:** {_hhmmss(job.duration_s)}")
    if job.speakers:
        display = ", ".join(names.get(s, s) for s in job.speakers)
        meta.append(f"**Partecipanti rilevati:** {display}")
    lines.append("  \n".join(meta))
    lines.append("")

    if a:
        if a.overview:
            lines += ["## Sintesi", "", a.overview, ""]

        if a.action_points:
            lines += ["## Action point", ""]
            lines.append("| | Azione | Responsabile | Scadenza |")
            lines.append("|---|---|---|---|")
            for ap in a.action_points:
                mark = _PRIORITY_MARK.get((ap.priority or "").lower(), "⚪")
                lines.append(
                    f"| {mark} | {_cell(ap.task)} | {_cell(ap.owner)} | {_cell(ap.due)} |"
                )
            lines += ["", "> 🔴 alta · 🟡 media · 🟢 bassa priorità", ""]

        if a.decisions:
            lines += ["## Decisioni prese", ""]
            for d in a.decisions:
                lines.append(f"- **{d.decision}**")
                if d.context:
                    lines.append(f"  - _{d.context}_")
            lines.append("")

        if a.sections:
            lines += ["## Argomenti trattati", ""]
            for s in a.sections:
                lines += [f"### {s.title}", "", s.content, ""]

        if a.open_questions:
            lines += ["## Questioni aperte", ""]
            lines += [f"- {q}" for q in a.open_questions]
            lines.append("")

    if job.overlap and job.overlap.count:
        o = job.overlap
        lines += [
            "## Parlato sovrapposto",
            "",
            f"{o.total_s}s ({o.percent}% della riunione) in {o.count} tratti, "
            f"fino a {o.max_speakers} voci contemporanee.",
            "",
            "> Le voci sono state separate in tracce distinte: ogni riga riporta "
            "ciò che ha detto davvero quella persona."
            if o.separated
            else "> In quei tratti la trascrizione riporta un solo flusso audio e "
            "l'attribuzione è incerta: le righe marcate ⚠ vanno verificate.",
            "",
        ]

    if include_transcript and job.segments:
        lines += ["---", "", "## Trascrizione integrale", ""]
        current_speaker = None

        # Le sovrapposizioni non separate arrivano qui come copie dello stesso
        # testo, una per voce: stampate così sembrano una frase ripetuta tre
        # volte. `group_for_reading` le ricuce in un evento unico.
        for group in overlap.group_for_reading(job.segments):
            stamp = _hhmmss(group["start"])

            if not group["overlapping"]:
                speaker = names.get(group["speakers"][0], group["speakers"][0])
                if speaker != current_speaker:
                    lines += ["", f"**{speaker}** _[{stamp}]_"]
                    current_speaker = speaker
                lines.append(group["texts"][0][1] if group["texts"] else "")
                continue

            who = " + ".join(names.get(s, s) for s in group["speakers"])
            current_speaker = None  # dopo un blocco l'intestazione va ripetuta

            if group["separated"] and len(group["texts"]) > 1:
                lines += ["", f"**{who}** _[{stamp}]_ — voci separate", ""]
                for speaker, text in group["texts"]:
                    lines.append(f"- **{names.get(speaker, speaker)}:** {text}")
            else:
                text = group["texts"][0][1] if group["texts"] else ""
                lines += [
                    "",
                    f"**{who}** _[{stamp}]_ — ⚠ parlano insieme",
                    "",
                    f"> {text}",
                    "",
                    "> _Whisper trascrive un solo flusso audio: questa è la frase "
                    "che ha distinto, non attribuibile con certezza a una delle voci._",
                ]
        lines.append("")

    return "\n".join(lines)


def _cell(value: str | None) -> str:
    """Neutralizza le pipe che romperebbero la tabella markdown."""
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()
