#!/usr/bin/env python3
"""Verifica della derivazione dei task dagli action point.

    python3 backend/tools/check_tasks.py

I due punti delicati:

* **non creare doppioni** quando il verbale viene rigenerato. Il modello
  riformula le stesse frasi con parole diverse, e senza un confronto per
  parole significative ogni rigenerazione moltiplicherebbe i task, ognuno con
  i propri commenti sparsi;
* **non inventare l'ancoraggio**. Il collegamento al punto della riunione è
  un'euristica sulle parole in comune: sotto una soglia netta è meglio nessun
  link che un link che porta nel punto sbagliato.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
spec = importlib.util.spec_from_file_location("tasks", APP / "tasks.py")
assert spec and spec.loader
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)

passed = 0
failed = 0


def check(name: str, cond: bool, detail: object = "") -> None:
    global passed, failed
    print(("  \u2713 " if cond else "  \u2717 ") + name + ("" if cond else f"  \u2192 {detail}"))
    passed += bool(cond)
    failed += not cond


class FakePoint:
    def __init__(self, task, owner="Non assegnato", due="", due_at="", priority="media"):
        self.task = task
        self.owner = owner
        self.due = due
        self.due_at = due_at
        self.due_precision = "giorno" if due_at else ""
        self.priority = priority


class FakeAnalysis:
    def __init__(self, points):
        self.action_points = points


SEGMENTS = [
    {"start": 0.0, "text": "buongiorno a tutti cominciamo"},
    {"start": 30.0, "text": "allora bisogna registrare un video dimostrativo "
                            "dell'applicazione entro domattina"},
    {"start": 90.0, "text": "poi vediamo i contatori dell'acqua e la vista di riepilogo"},
]

print("\nRiconoscere lo stesso impegno riformulato")
check("stessa frase", tasks.same_task("Preparare il preventivo", "Preparare il preventivo"))
check("frase estesa dal modello",
      tasks.same_task("Preparare il preventivo",
                      "Preparare il preventivo definitivo per il cliente Acme"))
check("ordine diverso",
      tasks.same_task("Registrare un video dimostrativo", "Video dimostrativo da registrare"))
check("accenti e maiuscole",
      tasks.same_task("Verificare la connessione", "VERIFICARE LA CONNESSIONE"))
check("impegni diversi NON coincidono",
      not tasks.same_task("Preparare il preventivo", "Chiamare il fornitore"))
check("argomento comune ma azione diversa",
      not tasks.same_task("Registrare il video dimostrativo",
                          "Pubblicare il listino prezzi aggiornato"))

print("\nAncoraggio al punto della riunione")
found = tasks.find_source_time("Registrare un video dimostrativo dell'applicazione mobile", SEGMENTS)
check(f"trova il segmento giusto ({found}s)", found == 30.0, found)
check("nessuna corrispondenza: nessun link inventato",
      tasks.find_source_time("Comprare una macchina nuova per l'ufficio", SEGMENTS) is None)
check("elenco vuoto", tasks.find_source_time("qualcosa", []) is None)
check("testo vuoto", tasks.find_source_time("", SEGMENTS) is None)

print("\nDerivazione dal verbale")
analysis = FakeAnalysis([
    FakePoint("Registrare un video dimostrativo", "Alessandro",
              "entro domattina", "2026-08-06T09:00"),
    FakePoint("Chiamare il fornitore"),
])
fresh = tasks.derive(analysis, SEGMENTS, "job-1", [])
check(f"un task per action point ({len(fresh)})", len(fresh) == 2)
check("porta con se la scadenza risolta", fresh[0]["due_at"] == "2026-08-06T09:00")
check("e la frase originale", fresh[0]["due_phrase"] == "entro domattina")
check("l'istante di origine e agganciato", fresh[0]["source_s"] == 30.0)
check("\u00abNon assegnato\u00bb diventa vuoto, non un nome finto", fresh[1]["owner"] == "")

print("\nRigenerazione: niente doppioni, niente lavoro perso")
existing = [{"job_id": "job-1", "title": "Registrare un video dimostrativo",
             "status": "in_corso", "notes": "chiesto a Marco"}]
again = tasks.derive(analysis, SEGMENTS, "job-1", existing)
check(f"il task gia presente non viene ricreato ({len(again)})", len(again) == 1)
check("e quello nuovo si", again[0]["title"] == "Chiamare il fornitore")
check("nemmeno se il modello lo riformula",
      len(tasks.derive(FakeAnalysis([FakePoint("Registrare il video dimostrativo dell'app mobile")]),
                       SEGMENTS, "job-1", existing)) == 0)
check("i task di un'altra riunione non interferiscono",
      len(tasks.derive(analysis, SEGMENTS, "job-2", existing)) == 2)
check("action point senza testo ignorati",
      len(tasks.derive(FakeAnalysis([FakePoint("   ")]), SEGMENTS, "job-9", [])) == 0)

print(f"\n{passed} superate, {failed} fallite")
sys.exit(1 if failed else 0)
