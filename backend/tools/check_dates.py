#!/usr/bin/env python3
"""Verifica della risoluzione delle scadenze dette a voce.

    python3 backend/tools/check_dates.py

L'ancora è sempre esplicita, mai l'ora corrente: un test sulle date che
dipende da quando lo esegui non è un test.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
spec = importlib.util.spec_from_file_location("dates", APP / "dates.py")
assert spec and spec.loader
dates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dates)

passed = 0
failed = 0

# Mercoledì 5 agosto 2026, 14:16 — la riunione vera che ha fatto nascere
# questo modulo, dove «entro domattina» era finito in «Non specificata».
ANCHOR = datetime(2026, 8, 5, 14, 16)


def check(phrase: str, expected_at: str, expected_precision: str = "") -> None:
    global passed, failed
    r = dates.resolve(phrase, ANCHOR)
    got = str(r["at"])[:16] if r["at"] else ""
    ok = got == expected_at and (not expected_precision or r["precision"] == expected_precision)
    label = f"{phrase!r:<34} → {got or '(nessuna)':<16}"
    print(("  ✓ " if ok else "  ✗ ") + label +
          ("" if ok else f"  atteso {expected_at or '(nessuna)'}"))
    passed += ok
    failed += not ok


print(f"\nAncora: mercoledì {ANCHOR:%d/%m/%Y %H:%M}\n")

print("Il caso che ha originato il modulo")
check("entro domattina", "2026-08-06T09:00", dates.PRECISION_TIME)
check("domattina", "2026-08-06T09:00")

print("\nGiorni relativi")
check("oggi", "2026-08-05T00:00", dates.PRECISION_DAY)
check("domani", "2026-08-06T00:00")
check("dopodomani", "2026-08-07T00:00")
check("entro domani sera", "2026-08-06T18:00", dates.PRECISION_TIME)
check("stasera", "2026-08-05T18:00")
check("stamattina", "2026-08-05T09:00")
check("in giornata", "2026-08-05T00:00")
check("subito", "2026-08-05T00:00")

print("\nGiorni della settimana (l'ancora è mercoledì)")
check("venerdi", "2026-08-07T00:00")
check("venerdì", "2026-08-07T00:00")          # con accento
check("entro venerdì", "2026-08-07T00:00")
check("lunedì", "2026-08-10T00:00")           # la settimana dopo
check("mercoledì", "2026-08-12T00:00")        # oggi è mercoledì: il prossimo
check("giovedì prossimo", "2026-08-06T00:00")
check("venerdì mattina", "2026-08-07T09:00", dates.PRECISION_TIME)

print("\nIntervalli")
check("fra due settimane", "2026-08-19T00:00")
check("entro tre giorni", "2026-08-08T00:00")
check("tra 10 giorni", "2026-08-15T00:00")
check("fra un mese", "2026-09-05T00:00")
check("entro una settimana", "2026-08-12T00:00")

print("\nDate esplicite")
check("il 15 settembre", "2026-09-15T00:00")
check("15 settembre 2027", "2027-09-15T00:00")
check("il 3 marzo", "2027-03-03T00:00")        # già passato: anno prossimo
check("15/09", "2026-09-15T00:00")
check("15-09-2026", "2026-09-15T00:00")
check("31/12/26", "2026-12-31T00:00")

print("\nGrana più larga")
check("fine settimana", "2026-08-08T00:00")    # il sabato
check("settimana prossima", "2026-08-10T00:00", dates.PRECISION_WEEK)
check("fine mese", "2026-08-31T00:00")
check("mese prossimo", "2026-09-01T00:00")

print("\nOre precise")
check("domani alle 15", "2026-08-06T15:00", dates.PRECISION_TIME)
check("venerdì alle 9:30", "2026-08-07T09:30")
check("alle 18", "2026-08-05T18:00")
check("domani alle 3", "2026-08-06T15:00")     # pomeriggio implicito

print("\nQuando NON deve inventare nulla")
check("Non specificata", "")
check("", "")
check("da definire", "")
check("appena avremo il preventivo", "")
check("quando sarà pronto", "")
check("prima o poi", "")

print("\nCasi limite")
check("il 31 febbraio", "")                    # data inesistente
check("il 45 gennaio", "")
check("fra settecento giorni", "")             # numero non riconosciuto
check("alle 99", "")

# Fine mese e mesi di lunghezza diversa
JAN31 = datetime(2026, 1, 31, 10, 0)
r = dates.resolve("fra un mese", JAN31)
ok = str(r["at"])[:10] == "2026-02-28"
print(("  ✓ " if ok else "  ✗ ") +
      f"{'31 gennaio + un mese':<34} → {str(r['at'])[:10]}  (non esiste il 31/02)")
passed += ok
failed += not ok

# Cambio d'anno
DEC30 = datetime(2026, 12, 30, 9, 0)
r = dates.resolve("dopodomani", DEC30)
ok = str(r["at"])[:10] == "2027-01-01"
print(("  ✓ " if ok else "  ✗ ") +
      f"{'30 dicembre + dopodomani':<34} → {str(r['at'])[:10]}  (cambio anno)")
passed += ok
failed += not ok

print("\nEtichette leggibili")
for iso, prec, atteso in [
    ("2026-08-06T09:00:00", dates.PRECISION_TIME, "06/08/2026 alle 09:00"),
    ("2026-08-06T00:00:00", dates.PRECISION_DAY, "06/08/2026"),
    ("2026-08-10T00:00:00", dates.PRECISION_WEEK, "settimana del 10/08/2026"),
    ("", "", "Non specificata"),
]:
    got = dates.describe(iso, prec)
    ok = got == atteso
    print(("  ✓ " if ok else "  ✗ ") + f"{got:<34}" + ("" if ok else f"  atteso {atteso}"))
    passed += ok
    failed += not ok

print(f"\n{passed} superate, {failed} fallite")
sys.exit(1 if failed else 0)
