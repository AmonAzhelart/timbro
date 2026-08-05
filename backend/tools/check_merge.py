#!/usr/bin/env python3
"""Verifica dell'unione fra trascrizione della miscela e tracce separate.

    python3 backend/tools/check_merge.py

Il punto delicato è che i due insiemi di segmenti descrivono lo stesso audio
con confini diversi. Deciderne la sorte per segmento intero, con un semplice
test di contatto, sbaglia in due modi opposti a seconda di dove cadono i
confini:

* il segmento della miscela NON tocca una sovrapposizione ma un segmento
  separato lungo la attraversa → lo stesso parlato compare due volte, con
  orari leggermente diversi e testo quasi identico;
* il segmento della miscela tocca una sovrapposizione anche solo per mezzo
  secondo → viene scartato per intero, e con lui decine di secondi di parlato
  pulito che nessuno recupera.

Sul primo job di prova il primo sintomo valeva 167 secondi coperti due volte
su 679. Questi controlli riproducono entrambi i casi.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
spec = importlib.util.spec_from_file_location("overlap", APP / "overlap.py")
assert spec and spec.loader
overlap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlap)

passed = 0
failed = 0


def check(name: str, cond: bool, detail: object = "") -> None:
    global passed, failed
    print(("  ✓ " if cond else "  ✗ ") + name + ("" if cond else f"  → {detail}"))
    passed += bool(cond)
    failed += not cond


def seg(t0: float, t1: float, speaker: str, text: str, separated: bool = False) -> dict:
    words = text.split()
    step = (t1 - t0) / len(words)
    return {
        "start": t0,
        "end": t1,
        "speaker": speaker,
        "text": text,
        **({"separated": True} if separated else {}),
        "words": [
            {"word": w, "start": round(t0 + i * step, 2), "end": round(t0 + (i + 1) * step, 2)}
            for i, w in enumerate(words)
        ],
    }


def merge_new(base: list[dict], separated: list[dict], overlaps: list[dict]) -> list[dict]:
    """Riproduce `transcribe._merge_separated` senza importarlo (userebbe pydantic)."""
    spans = overlap.overlap_spans(overlaps)
    base_pieces = overlap.split_at_boundaries(base, overlaps)
    sep_pieces = overlap.split_at_boundaries(separated, overlaps)

    replaced = [s for s in sep_pieces if overlap.fraction_inside(s, spans) >= 0.5]
    taken = overlap.merge_spans([(s["start"], s["end"]) for s in replaced])

    kept = [
        p for p in base_pieces
        if not (overlap.fraction_inside(p, spans) >= 0.5
                and overlap.fraction_inside(p, taken) >= 0.5)
    ]
    return sorted(kept + replaced, key=lambda s: (s["start"], s["speaker"]))


def coverage(segments: list[dict]) -> float:
    return overlap.covered_seconds([(s["start"], s["end"]) for s in segments])


def merge_old(base: list[dict], separated: list[dict], overlaps: list[dict]) -> list[dict]:
    """La logica precedente, tenuta come termine di paragone."""
    def touches(s: dict) -> bool:
        return any(o["start"] < s["end"] and o["end"] > s["start"] for o in overlaps)

    return sorted(
        [s for s in base if not touches(s)] + [s for s in separated if touches(s)],
        key=lambda s: (s["start"], s["speaker"]),
    )


def duplicated_seconds(segments: list[dict]) -> float:
    """Secondi in cui la STESSA voce compare due volte.

    Due voci diverse nello stesso istante sono legittime — è la definizione di
    sovrapposizione. Solo lo stesso parlante due volte è un difetto.
    """
    ordered = sorted(segments, key=lambda s: (s["speaker"], s["start"]))
    total = 0.0
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b["speaker"] != a["speaker"] or b["start"] >= a["end"]:
                break
            total += min(a["end"], b["end"]) - b["start"]
    return total


def spoken_words(segments: list[dict]) -> set[str]:
    return {w for s in segments for w in s["text"].split()}


# ---------------------------------------------------------------------------
# Caso 1 — segmento della miscela che NON tocca: nasce la duplicazione
# ---------------------------------------------------------------------------
print("\nCaso 1 — duplicazione: il parlato compare due volte")
BASE_1 = [
    seg(0.0, 13.0, "SPEAKER_00", "allora io so che il flipper ce lha anche coso salvatore"),
    seg(23.0, 30.0, "SPEAKER_00", "in teoria se io ora me ne vado qua"),
]
SEP_1 = [
    seg(8.0, 26.0, "SPEAKER_00",
        "io so che il flipper ce lha anche coso salvatore se lera comprato", separated=True),
    seg(8.0, 26.0, "SPEAKER_01",
        "ora lo compriamo e facciamo vedeteci un testing prendere i segnali", separated=True),
]
OVL_1 = [{"start": 14.0, "end": 22.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]

old_1, new_1 = merge_old(BASE_1, SEP_1, OVL_1), merge_new(BASE_1, SEP_1, OVL_1)
print(f"    precedente: {duplicated_seconds(old_1):5.2f}s della stessa voce due volte")
print(f"    corretto:   {duplicated_seconds(new_1):5.2f}s")
check("la logica precedente duplicava", duplicated_seconds(old_1) > 3)
check("la nuova non duplica", duplicated_seconds(new_1) < 1.0, duplicated_seconds(new_1))

# ---------------------------------------------------------------------------
# Caso 2 — segmento della miscela che tocca appena: nasce la perdita
# ---------------------------------------------------------------------------
print("\nCaso 2 — perdita: il parlato pulito viene buttato")
BASE_2 = [
    seg(0.03, 22.59, "SPEAKER_00",
        "parliamo di qualcosaltro perche e sempre qua allora stavo dicendo comunque "
        "e figo io so che il flipper ce lha anche coso salvatore"),
]
SEP_2 = [
    seg(14.0, 22.0, "SPEAKER_00", "io so che il flipper", separated=True),
    seg(14.0, 22.0, "SPEAKER_01", "ora lo compriamo e facciamo", separated=True),
]
OVL_2 = [{"start": 14.9, "end": 22.4, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]

old_2, new_2 = merge_old(BASE_2, SEP_2, OVL_2), merge_new(BASE_2, SEP_2, OVL_2)
lost_old = spoken_words(BASE_2) - spoken_words(old_2)
lost_new = spoken_words(BASE_2) - spoken_words(new_2)
print(f"    precedente: {len(lost_old)} parole della miscela perdute")
print(f"    corretto:   {len(lost_new)}")
check("la logica precedente perdeva il parlato pulito", len(lost_old) > 5, lost_old)
check("la nuova conserva l'inizio dell'intervento",
      "parliamo" in spoken_words(new_2) and "qualcosaltro" in spoken_words(new_2))

# ---------------------------------------------------------------------------
# Caso 3 — provenienza e marcatura
# ---------------------------------------------------------------------------
print("\nCaso 3 — provenienza dei pezzi e marcatura per l'interfaccia")
check("dentro la sovrapposizione: solo tracce separate",
      all(s.get("separated") for s in new_1 if overlap.fraction_inside(s, overlap.overlap_spans(OVL_1)) >= 0.5))
check("fuori: solo trascrizione della miscela",
      all(not s.get("separated") for s in new_1 if not overlap.fraction_inside(s, overlap.overlap_spans(OVL_1)) >= 0.5))

marked = overlap.annotate([dict(s) for s in new_1], OVL_1, duplicate=False)
n_marked = sum(1 for s in marked if len(s.get("overlap") or []) >= 2)
check(f"i segmenti sovrapposti sono marcati ({n_marked})", n_marked >= 2, n_marked)
check("con duplicate=False il testo non viene replicato", len(marked) == len(new_1))

groups = [g for g in overlap.group_for_reading(marked) if g["overlapping"]]
check("l'interfaccia li raggruppa in un evento simultaneo", len(groups) == 1, len(groups))
check("con due voci e due testi distinti",
      bool(groups) and len(groups[0]["speakers"]) == 2 and len(groups[0]["texts"]) == 2)

# ---------------------------------------------------------------------------
# Caso 4 — casi limite
# ---------------------------------------------------------------------------
print("\nCaso 4 — casi limite")
check("nessuna sovrapposizione: segmenti intatti",
      overlap.split_at_boundaries(BASE_1, []) == BASE_1)
check("senza tempi di parola il segmento resta intero",
      len(overlap.split_at_boundaries(
          [{"start": 0.0, "end": 10.0, "speaker": "A", "text": "x"}], OVL_1)) == 1)
check("elenco vuoto", overlap.split_at_boundaries([], OVL_1) == [])
check("i pezzi restano ordinati nel tempo",
      new_1 == sorted(new_1, key=lambda s: (s["start"], s["speaker"])))

# ---------------------------------------------------------------------------
# Caso 5 — il caso reale: le tracce separate non hanno i tempi di parola
# ---------------------------------------------------------------------------
print("\nCaso 5 — tracce separate senza tempi di parola (come le produce Whisper)")


def wordless(t0: float, t1: float, speaker: str, text: str) -> dict:
    """Come `_transcribe_separated`: niente allineamento, niente `words`."""
    return {"start": t0, "end": t1, "speaker": speaker, "text": text, "separated": True}


BASE_5 = [
    seg(0.0, 22.0, "SPEAKER_00",
        "parliamo di qualcosaltro perche e sempre qua allora stavo dicendo comunque "
        "e figo io so che il flipper ce lha anche coso salvatore mazzaglia"),
    seg(24.0, 46.0, "SPEAKER_00",
        "in teoria se io ora me ne vado qua e avro il terminale poi vediamo "
        "cosa succede quando lo colleghiamo alla rete di casa"),
]
# Un solo segmento separato, lungo, che attraversa più sovrapposizioni.
SEP_5 = [wordless(8.0, 34.0, "SPEAKER_01",
                  "ora lo compriamo e facciamo vedeteci un testing prendere i segnali")]
# Lo stesso, ma allineato: è ciò che si ottiene ora che le tracce separate
# passano dall'allineamento come la miscela. La densità è quella reale —
# nella riunione di prova le parole durano circa 0.4s e le sovrapposizioni
# circa 1.07s, quindi le parole sono più fini delle finestre e il taglio è
# possibile. Con parole più lunghe delle finestre non lo sarebbe, e la
# fusione giustamente preferisce la miscela.
_PAROLE_5 = ("ora lo compriamo e facciamo vedeteci un testing vediamo se qua dentro "
             "riusciamo a prendere tutti i segnali tante che ci vogliono due secondi "
             "perche ho capito allora perche tutto fa schifo quando facevano "
             "ingazzare io rompevo tutte le cose allora ho capito certo tu dici "
             "voleva fare cose guarda prima cosa sai sono cattolico ora lo "
             "lasciamo da qui e tutto registrato eh perche")
SEP_5_ALIGNED = [seg(8.0, 34.0, "SPEAKER_01", _PAROLE_5, separated=True)]
# Sovrapposizioni brevi e numerose, come nella riunione vera (~1s l'una)
OVL_5 = [{"start": t, "end": t + 1.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]}
         for t in (10.0, 14.0, 18.0, 26.0, 30.0, 40.0)]

new_5 = merge_new(BASE_5, SEP_5, OVL_5)
cov_base, cov_new = coverage(BASE_5), coverage(new_5)
print(f"    copertura base {cov_base:.1f}s → risultato {cov_new:.1f}s")
check("la fusione non riduce l'audio coperto da testo", cov_new >= cov_base - 1.0,
      f"persi {cov_base - cov_new:.1f}s")
check("il parlato iniziale non sparisce", "parliamo" in spoken_words(new_5))
check("né quello finale", "terminale" in spoken_words(new_5))
check("senza allineamento le tracce separate restano inutilizzabili",
      not any(s.get("separated") for s in new_5),
      "sarebbero entrate intere, duplicando la miscela")

# Con l'allineamento il segmento si taglia e le sue parti utili entrano.
new_5a = merge_new(BASE_5, SEP_5_ALIGNED, OVL_5)
cov_5a = coverage(new_5a)
usati = sum(1 for s in new_5a if s.get("separated"))
print(f"    con allineamento: copertura {cov_5a:.1f}s, {usati} pezzi separati usati")
check("con l'allineamento le tracce separate vengono usate", usati > 0, usati)
# Tagliare sui confini di parola costa qualche decimo di secondo per confine:
# la soglia è proporzionale, come nel codice di produzione.
check(f"la copertura resta entro il 5% ({(1-cov_5a/cov_base)*100:.1f}% persi)",
      cov_5a >= cov_base * 0.95 - 1.0, f"persi {cov_base - cov_5a:.1f}s")
check("il parlato della miscela resta comunque integro",
      "parliamo" in spoken_words(new_5a) and "terminale" in spoken_words(new_5a))
check("i pezzi separati usati stanno dentro le sovrapposizioni",
      all(overlap.fraction_inside(s, overlap.overlap_spans(OVL_5)) >= 0.5
          for s in new_5a if s.get("separated")))
check("nessun tratto della stessa voce compare due volte",
      duplicated_seconds(new_5a) < 1.0, duplicated_seconds(new_5a))

# Con il test sul punto centrale (la versione che ha causato i buchi)
mid_kept = [p for p in overlap.split_at_boundaries(BASE_5, OVL_5)
            if not any(o["start"] <= (p["start"] + p["end"]) / 2 < o["end"] for o in OVL_5)]
mid_repl = [s for s in SEP_5
            if any(o["start"] <= (s["start"] + s["end"]) / 2 < o["end"] for o in OVL_5)]
cov_mid = coverage(mid_kept + mid_repl)
print(f"    (col punto centrale: {cov_mid:.1f}s, {cov_base - cov_mid:.1f}s persi)")
check("la versione col punto centrale perdeva davvero parlato", cov_mid < cov_base - 1.0)
check("e scartava quasi tutte le tracce separate", len(mid_repl) == 0, len(mid_repl))

# ---------------------------------------------------------------------------
# Caso 6 — la rete di sicurezza deve scattare sul crollo, non sugli scarti
# ---------------------------------------------------------------------------
print("\nCaso 6 — rete di sicurezza sulla copertura")


def scatta(before: float, after: float) -> bool:
    """Stessa condizione del codice di produzione."""
    return after < before * 0.95 - 1.0


check("non scatta sugli scarti ai bordi (44.0 → 42.9)", not scatta(44.0, 42.9))
check("non scatta su una perdita dell'1% (600 → 594)", not scatta(600.0, 594.0))
check("scatta sul crollo osservato (679 → 433, -36%)", scatta(679.0, 433.0))
check("scatta su una perdita del 20%", scatta(600.0, 480.0))
check("regge la copertura nulla", scatta(600.0, 0.0))
check("non scatta se non c'è nulla da coprire", not scatta(0.0, 0.0))

print(f"\n{passed} superate, {failed} fallite")
sys.exit(1 if failed else 0)
