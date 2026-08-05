#!/usr/bin/env python3
"""Quanto parlato è rimasto fuori dalla trascrizione, misurato sull'audio.

    docker compose exec api python tools/check_coverage.py            # ultimo job
    docker compose exec api python tools/check_coverage.py <job_id>

Nasce da una domanda ricorrente — «mi sembra che manchi della roba» — a cui
finora si poteva rispondere solo a impressione. Guardare la percentuale di
audio senza testo non basta: in una conversazione le pause valgono
tranquillamente un quarto della durata, e il 74% di copertura può essere
perfetto o pessimo a seconda di dove cadono i buchi.

Il metodo: per ogni tratto senza testo si misura quanta parte sta sopra la
soglia di energia del parlato. Le pause vere stanno sotto il 10%; i tratti
trascritti stanno intorno al 90%. Ciò che sta in mezzo è parlato perso, e
va contato in secondi invece che discusso.

Serve ffmpeg, che nel container c'è già.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/data"))
#: Sopra questa energia il segnale è compatibile col parlato. Sotto, è
#: rumore d'ambiente o silenzio.
SOGLIA_DB = -30
#: Frazione oltre la quale un buco è "sospetto": sopra un terzo del tempo con
#: energia da parlato, quel silenzio non è un silenzio.
SOSPETTO = 0.35
#: Sotto questa durata un buco è respiro fra due frasi, non un problema.
MIN_BUCO_S = 1.2


def gaps(segments: list[dict], duration: float) -> list[tuple[float, float]]:
    covered: list[list[float]] = []
    for seg in sorted(segments, key=lambda s: s["start"]):
        if covered and seg["start"] <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1], seg["end"])
        else:
            covered.append([seg["start"], seg["end"]])

    out: list[tuple[float, float]] = []
    previous = 0.0
    for start, end in covered:
        if start - previous > MIN_BUCO_S:
            out.append((previous, start))
        previous = end
    if duration - previous > MIN_BUCO_S:
        out.append((previous, duration))
    return out


def speech_fraction(audio: Path, start: float, end: float) -> float:
    """Quanta parte del tratto ha energia da parlato."""
    span = max(0.3, end - start)
    result = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", f"{start}", "-t", f"{span}", "-i", str(audio),
         "-af", f"silencedetect=noise={SOGLIA_DB}dB:d=0.25", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    silence = sum(float(x) for x in re.findall(r"silence_duration:\s*([\d.]+)", result.stdout))
    return max(0.0, min(1.0, 1 - silence / span))


def main() -> int:
    db = DATA / "timbro.sqlite3"
    if not db.exists():
        print(f"Database non trovato in {db}")
        return 1

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    if len(sys.argv) > 1:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id LIKE ?", (sys.argv[1] + "%",)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='done' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

    if row is None:
        print("Nessun job da analizzare.")
        return 1

    audio = Path(row["audio_path"] or "")
    if not audio.exists():
        print(f"Audio non disponibile ({audio}). "
              "Con «Elimina l'audio dopo l'elaborazione» attivo non è verificabile.")
        return 1

    segments = json.loads(row["segments"] or "[]")
    duration = row["duration_s"] or 0
    if not segments or not duration:
        print("Trascrizione vuota.")
        return 1

    covered = duration - sum(b - a for a, b in gaps(segments, duration)) - 0.0
    holes = gaps(segments, duration)

    print(f"\n{row['filename']} · {duration:.0f}s · {len(segments)} segmenti")
    print(f"audio senza testo: {sum(b - a for a, b in holes):.0f}s in {len(holes)} tratti\n")

    # Riferimento: quanto vale la frazione sui tratti che HANNO testo. Senza
    # un termine di paragone preso dallo stesso audio, la soglia sarebbe
    # arbitraria e cambierebbe con la qualità della registrazione.
    sample = segments[len(segments) // 4: len(segments) // 4 + 6] or segments[:6]
    reference = [speech_fraction(audio, s["start"], s["end"]) for s in sample]
    media_ref = sum(reference) / len(reference) if reference else 1.0

    sospetti: list[tuple[float, float, float]] = []
    print("tratti senza testo (frazione con energia da parlato):")
    for start, end in holes:
        fraction = speech_fraction(audio, start, end)
        if fraction >= SOSPETTO:
            sospetti.append((start, end, fraction))
        if end - start >= 2.0 or fraction >= SOSPETTO:
            mark = "  ← parlato non trascritto" if fraction >= SOSPETTO else ""
            print(f"   {start:7.1f}→{end:7.1f} ({end - start:5.1f}s)  {fraction * 100:5.1f}%{mark}")

    persi = sum(end - start for start, end, _ in sospetti)
    print(f"\n   riferimento sui tratti CON testo: {media_ref * 100:.0f}%")
    print(f"   pause vere:      {len(holes) - len(sospetti)} tratti")
    print(f"   parlato perso:   {len(sospetti)} tratti, {persi:.0f}s "
          f"({persi / duration * 100:.1f}% della riunione)")

    if persi / duration > 0.05:
        print("\n   Oltre il 5%: vale la pena intervenire.")
        print("   In ⚙ Impostazioni → Qualità della trascrizione:")
        print("   · abbassa «Sensibilità inizio parlato» (vad_onset) a 0.35")
        print("   · abbassa «Soglia di silenzio» (no_speech_threshold) a 0.4")
        print("   · attiva «Pulisci l'audio prima di trascrivere» se la "
              "registrazione è ambientale")
    elif sospetti:
        print("\n   Sotto il 5%: sono casi isolati, di norma voci molto basse "
              "o rumore di inizio registrazione.")
    else:
        print("\n   Nessun parlato perso: i buchi sono tutti pause reali.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
