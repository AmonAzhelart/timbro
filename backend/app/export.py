"""Esportazione del verbale in Markdown."""

from __future__ import annotations

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
        for seg in job.segments:
            speaker = names.get(seg.speaker, seg.speaker)
            overlapping = len(seg.overlap) > 1 and not seg.separated
            if speaker != current_speaker:
                lines.append("")
                mark = " ⚠" if overlapping else ""
                lines.append(f"**{speaker}**{mark} _[{_hhmmss(seg.start)}]_")
                current_speaker = speaker
            lines.append(seg.text)
        lines.append("")

    return "\n".join(lines)


def _cell(value: str | None) -> str:
    """Neutralizza le pipe che romperebbero la tabella markdown."""
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()
