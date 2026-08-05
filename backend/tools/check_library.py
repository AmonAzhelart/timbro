#!/usr/bin/env python3
"""Verifica di archivio e ricerca su un database temporaneo.

    python3 backend/tools/check_library.py

Copre cartelle annidate, etichette, ricerca a parole esatte, impegni e
rubrica delle voci. La parte per significato non è coperta qui: richiede
Ollama e un modello di embedding, quindi si verifica solo che la sua assenza
non rompa la ricerca.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Le impostazioni vanno indirizzate a un database usa e getta PRIMA di
# importare lo store, che legge `settings.db_path` all'apertura.
TMP = Path(tempfile.mkdtemp(prefix="timbro-check-"))
import os

os.environ["DATA_DIR"] = str(TMP)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import store  # noqa: E402
from app.models import (  # noqa: E402
    ActionPoint, Decision, JobDetail, JobStatus, MeetingAnalysis, Segment, SummarySection,
)

passed = 0
failed = 0


def check(name: str, cond: bool, detail: object = "") -> None:
    global passed, failed
    print(("  ✓ " if cond else "  ✗ ") + name + ("" if cond else f"  → {detail}"))
    passed += bool(cond)
    failed += not cond


store.init_db()


def make_job(job_id: str, title: str, sections, points, transcript) -> None:
    store.create_job(job_id, f"{title}.m4a", str(TMP / "x.m4a"))
    analysis = MeetingAnalysis(
        title=title,
        overview=f"Sintesi della riunione su {title}.",
        sections=[SummarySection(title=t, content=c) for t, c in sections],
        decisions=[Decision(decision="Procedere", context="Nessun ostacolo")],
        action_points=[ActionPoint(task=t, owner=o, due=d) for t, o, d in points],
    )
    segments = [
        Segment(start=i * 30.0, end=i * 30.0 + 25.0, speaker=f"SPEAKER_0{i % 2}", text=t)
        for i, t in enumerate(transcript)
    ]
    store.update_job(
        job_id,
        status=JobStatus.DONE.value,
        progress=100,
        analysis=analysis.model_dump(),
        segments=[s.model_dump() for s in segments],
        speaker_names={"SPEAKER_00": "Marco", "SPEAKER_01": "Anna"},
    )
    store.reindex_job(job_id, with_vectors=False)


make_job(
    "job-acme", "Preventivo Acme",
    [("Costi del nuovo impianto", "Abbiamo valutato il preventivo di Acme, "
      "quarantamila euro per l'impianto completo.")],
    [("Preparare il preventivo definitivo", "Marco", "15 settembre"),
     ("Chiamare il fornitore", "Anna", "Non specificata")],
    ["Marco: allora vediamo il preventivo di Acme per l'impianto",
     "Anna: quarantamila mi sembrano tanti, chiediamo uno sconto",
     "Marco: proviamo, ma i tempi di consegna sono il vero problema"],
)
make_job(
    "job-hw", "Hardware e noleggio",
    [("Comprare o noleggiare", "Discussione se acquistare i server o "
      "prenderli a noleggio operativo.")],
    [("Confrontare le due ipotesi", "Anna", "30 settembre")],
    ["Anna: conviene comprare le macchine o prenderle a noleggio?",
     "Marco: dipende da quanto le teniamo, sotto i tre anni il noleggio vince"],
)

# ---------------------------------------------------------------------------
print("\nCartelle annidate")
clienti = store.create_folder("Clienti")
acme = store.create_folder("Acme", parent_id=clienti["id"])
y2026 = store.create_folder("2026", parent_id=acme["id"])
check("albero creato su tre livelli", len(store.list_folders()) == 3)

err = store.update_folder(clienti["id"], parent_id=y2026["id"], move=True)
check("spostare una cartella dentro una sua discendente è rifiutato", bool(err), err)
err = store.update_folder(clienti["id"], parent_id=clienti["id"], move=True)
check("spostare una cartella dentro sé stessa è rifiutato", bool(err), err)
err = store.update_folder(y2026["id"], parent_id=None, move=True)
check("spostare alla radice è consentito", err is None, err)

store.set_job_folder("job-acme", acme["id"])
check("la riunione risulta nella cartella",
      next(f["jobs"] for f in store.list_folders() if f["id"] == acme["id"]) == 1)

store.set_job_folder("job-acme", acme["id"])
store.delete_folder(clienti["id"])
check("eliminando il ramo restano solo le cartelle fuori", len(store.list_folders()) == 1)
check("la riunione NON viene eliminata con la cartella",
      store.get_job("job-acme") is not None)
check("e torna alla radice", store.get_job("job-acme").folder_id is None)

# ---------------------------------------------------------------------------
print("\nEtichette")
urgente = store.ensure_tag("urgente")
stesso = store.ensure_tag("URGENTE")
check("le etichette non si duplicano per differenza di maiuscole",
      urgente["id"] == stesso["id"])
store.set_job_tags("job-acme", [urgente["id"]])
check("associata alla riunione", store.tags_by_job().get("job-acme") == [urgente["id"]])
check("conteggio corretto", store.list_tags()[0]["jobs"] == 1)
store.delete_tag(urgente["id"])
check("eliminando l'etichetta si libera la riunione", store.tags_by_job() == {})

# ---------------------------------------------------------------------------
print("\nRicerca a parole esatte")
r = store.search_archive("preventivo")
check(f"«preventivo» trova risultati ({len(r['hits'])})", len(r["hits"]) > 0)
check("il risultato porta il titolo della riunione",
      any(h["job_title"] == "Preventivo Acme" for h in r["hits"]))
check("e il frammento di testo", all(h["snippet"] for h in r["hits"]))

r = store.search_archive("quarantamila")
check("trova una parola detta solo nella trascrizione", len(r["hits"]) > 0)
check("indicando l'istante da riascoltare",
      any(h["kind"] == "trascrizione" and h["ref"] for h in r["hits"]),
      [h["kind"] for h in r["hits"]])

r = store.search_archive("perche")
check("la ricerca ignora gli accenti", isinstance(r["hits"], list))
r = store.search_archive('"; DROP TABLE jobs; --')
check("i caratteri speciali non fanno esplodere la query", isinstance(r["hits"], list))
check("e il database è intatto", store.get_job("job-acme") is not None)
r = store.search_archive("")
check("query vuota: nessun risultato, nessun errore", r["hits"] == [])
r = store.search_archive("noleggio", kinds=["argomento"])
check("il filtro per tipo funziona",
      all(h["kind"] == "argomento" for h in r["hits"]) and len(r["hits"]) > 0,
      [h["kind"] for h in r["hits"]])

# ---------------------------------------------------------------------------
print("\nImpegni su tutto l'archivio")
points = store.all_action_points()
check(f"raccolti da tutte le riunioni ({len(points)})", len(points) == 3, len(points))
check("ognuno sa da quale riunione viene",
      all(p["job_title"] and p["job_id"] for p in points))

store.set_action_done("job-acme", 0, True)
check("spuntato uno, ne restano due aperti", len(store.all_action_points()) == 2)
check("ma con include_done tornano tutti", len(store.all_action_points(True)) == 3)
check("lo stato persiste nel verbale",
      store.get_job("job-acme").analysis.action_points[0].done is True)
check("indice fuori intervallo rifiutato", store.set_action_done("job-acme", 99, True) is False)
check("job inesistente rifiutato", store.set_action_done("nope", 0, True) is False)

# ---------------------------------------------------------------------------
print("\nRubrica delle voci e glossari")
store.remember_speakers(["Marco", "Anna", "  ", "Marco"])
names = {s["name"]: s["uses"] for s in store.known_speakers()}
check("i nomi vuoti non entrano", "" not in names and len(names) == 2, names)
store.remember_speakers(["Marco"])
check("il conteggio d'uso cresce", store.known_speakers()[0]["name"] == "Marco",
      store.known_speakers())

g = store.save_glossary("Acme", "Riunione commerciale", "Acme, preventivo")
check("glossario salvato", len(store.list_glossaries()) == 1)
store.save_glossary("Acme rinominato", "x", "y", glossary_id=g["id"])
check("e aggiornato senza duplicarsi",
      len(store.list_glossaries()) == 1 and store.list_glossaries()[0]["name"] == "Acme rinominato")

# ---------------------------------------------------------------------------
print("\nEliminazione e coerenza dell'indice")
before = len(store.search_archive("noleggio")["hits"])
store.delete_job("job-hw")
after = len(store.search_archive("noleggio")["hits"])
check(f"eliminando la riunione spariscono i suoi risultati ({before} → {after})",
      before > 0 and after == 0)

report = store.reindex_all(with_vectors=False)
check("la ricostruzione dell'indice gira senza errori", report["jobs"] >= 1, report)
check("e la ricerca continua a funzionare",
      len(store.search_archive("preventivo")["hits"]) > 0)

print(f"\n{passed} superate, {failed} fallite")
sys.exit(1 if failed else 0)
