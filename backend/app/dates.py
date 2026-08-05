"""Risoluzione delle scadenze dette a voce in date vere.

In riunione nessuno dice «entro il 6 agosto 2026»: dice «entro domattina»,
«venerdì prossimo», «fra un paio di settimane». Finora quelle espressioni
finivano nel verbale come `Non specificata`, perché al modello era vietato
inventare date e senza un ancoraggio temporale non poteva fare altro.

La divisione dei compiti qui è netta e voluta:

* il **modello** riporta la frase esattamente come è stata pronunciata —
  è comprensione del linguaggio, ed è ciò che sa fare;
* **questo modulo** fa l'aritmetica sul calendario partendo dall'istante
  della riunione.

Il contrario — chiedere all'LLM di calcolare «tre giorni dopo il 5 agosto» —
è il modo più affidabile di ottenere date sbagliate senza accorgersene, e
soprattutto non verificabile: qui invece ogni regola ha una prova.

Tutto è in italiano perché è la lingua delle riunioni trascritte; l'ancora è
sempre esplicita, mai `datetime.now()`, così i test sono deterministici.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date, datetime, time, timedelta

log = logging.getLogger(__name__)

#: Precisione della scadenza. Serve all'interfaccia per non mostrare
#: un'ora inventata quando è stato detto solo un giorno.
PRECISION_NONE = ""
PRECISION_WEEK = "settimana"
PRECISION_DAY = "giorno"
PRECISION_TIME = "ora"

GIORNI = {
    "lunedi": 0, "martedi": 1, "mercoledi": 2, "giovedi": 3,
    "venerdi": 4, "sabato": 5, "domenica": 6,
}
MESI = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5,
    "giugno": 6, "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10,
    "novembre": 11, "dicembre": 12,
}
NUMERI = {
    "un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
    "sei": 6, "sette": 7, "otto": 8, "nove": 9, "dieci": 10, "undici": 11,
    "dodici": 12, "quindici": 15, "venti": 20, "trenta": 30,
}
#: Momento della giornata -> ora convenzionale. Sono convenzioni dichiarate,
#: non misure: «mattina» non significa le 9:00, ma bisogna pur scegliere.
MOMENTI = {
    "mattina": time(9, 0), "mattino": time(9, 0), "mattinata": time(9, 0),
    "pranzo": time(13, 0), "mezzogiorno": time(12, 0),
    "pomeriggio": time(15, 0), "sera": time(18, 0), "serata": time(18, 0),
    "notte": time(23, 0), "cena": time(20, 0),
}

NON_SPECIFICATA = {"", "non specificata", "non specificato", "nessuna",
                   "da definire", "non definita", "-", "—", "n/a"}


def _normalize(text: str) -> str:
    """Minuscole senza accenti: «Venerdì» e «venerdi» devono coincidere."""
    lowered = (text or "").strip().lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _quantity(word: str) -> int | None:
    if word.isdigit():
        return int(word)
    return NUMERI.get(word)


def _at(day: date, moment: time | None) -> tuple[datetime, str]:
    if moment is None:
        return datetime.combine(day, time(0, 0)), PRECISION_DAY
    return datetime.combine(day, moment), PRECISION_TIME


def resolve(phrase: str, anchor: datetime) -> dict[str, object]:
    """Da «entro domattina» + istante della riunione a una data vera.

    Restituisce ``{"at": ISO 8601 o "", "precision": ..., "phrase": ...}``.
    Quando non riconosce nulla restituisce `at` vuoto: meglio nessuna data che
    una data sbagliata, perché una scadenza inventata è peggio di una mancante.
    """
    empty = {"at": "", "precision": PRECISION_NONE, "phrase": (phrase or "").strip()}
    text = _normalize(phrase)
    if text in NON_SPECIFICATA:
        return empty

    today = anchor.date()

    # Momento della giornata citato esplicitamente, o dedotto dal suffisso
    # («domattina» = domani + mattina).
    moment: time | None = None
    for name, value in MOMENTI.items():
        if re.search(rf"\b{name}\b", text):
            moment = value
            break

    # Ora precisa: «alle 15», «alle 15:30», «per le 9».
    hour_match = re.search(r"\b(?:alle|per le|entro le|ore)\s+(\d{1,2})(?:[:.](\d{2}))?\b", text)
    if hour_match:
        hh = int(hour_match.group(1))
        mm = int(hour_match.group(2) or 0)
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            # «alle 3» in una riunione pomeridiana significa le 15.
            if hh < 8 and "mattina" not in text and "notte" not in text:
                hh += 12
            moment = time(hh, mm)

    # --- Forme contratte: domattina, stamattina, stasera, stanotte ---
    if re.search(r"\bdomattina\b", text):
        return {**empty, **_zip(_at(today + timedelta(days=1), moment or MOMENTI["mattina"]))}
    if re.search(r"\bstamattina\b|\bstamane\b", text):
        return {**empty, **_zip(_at(today, moment or MOMENTI["mattina"]))}
    if re.search(r"\bstasera\b", text):
        return {**empty, **_zip(_at(today, moment or MOMENTI["sera"]))}
    if re.search(r"\bstanotte\b", text):
        return {**empty, **_zip(_at(today, moment or MOMENTI["notte"]))}

    # --- Giorni relativi ---
    if re.search(r"\bdopodomani\b", text):
        return {**empty, **_zip(_at(today + timedelta(days=2), moment))}
    if re.search(r"\bdomani\b", text):
        return {**empty, **_zip(_at(today + timedelta(days=1), moment))}
    if re.search(r"\boggi\b|\bin giornata\b|\bentro oggi\b|\bsubito\b|\bappena possibile\b", text):
        return {**empty, **_zip(_at(today, moment))}
    if re.search(r"\bieri\b", text):
        return {**empty, **_zip(_at(today - timedelta(days=1), moment))}

    # --- «fra due settimane», «entro tre giorni», «fra un mese» ---
    delta = re.search(
        r"\b(?:fra|tra|entro|dopo)\s+(\w+)\s+(giorni|giorno|settimane|settimana|mesi|mese)\b",
        text,
    )
    if delta:
        n = _quantity(delta.group(1))
        if n is not None:
            unit = delta.group(2)
            if unit.startswith("giorn"):
                day = today + timedelta(days=n)
            elif unit.startswith("settiman"):
                day = today + timedelta(weeks=n)
            else:
                day = _add_months(today, n)
            return {**empty, **_zip(_at(day, moment))}

    # --- Giorno della settimana, con o senza «prossimo» ---
    weekday = re.search(r"\b(lunedi|martedi|mercoledi|giovedi|venerdi|sabato|domenica)\b", text)
    if weekday:
        target = GIORNI[weekday.group(1)]
        ahead = (target - today.weekday()) % 7
        # Oggi stesso non è "venerdì prossimo": si intende il successivo.
        if ahead == 0:
            ahead = 7
        if re.search(r"\bprossim|\bche viene\b", text):
            # «venerdì prossimo» detto di lunedì = questa settimana comunque;
            # detto di venerdì = fra sette giorni. `ahead` lo copre già.
            pass
        if re.search(r"\bscorso\b|\bpassato\b", text):
            ahead -= 7
        return {**empty, **_zip(_at(today + timedelta(days=ahead), moment))}

    # --- Data esplicita: «il 15 settembre», «15/09», «15-09-2026» ---
    explicit = re.search(r"\b(\d{1,2})\s+(" + "|".join(MESI) + r")\b(?:\s+(\d{4}))?", text)
    if explicit:
        day_n, month_name, year_s = explicit.groups()
        year = int(year_s) if year_s else today.year
        try:
            day = date(year, MESI[month_name], int(day_n))
        except ValueError:
            return empty
        # Un mese già passato senza anno indicato è quasi sempre l'anno dopo.
        if not year_s and day < today:
            try:
                day = day.replace(year=year + 1)
            except ValueError:
                return empty
        return {**empty, **_zip(_at(day, moment))}

    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", text)
    if numeric:
        d_s, m_s, y_s = numeric.groups()
        year = today.year
        if y_s:
            year = int(y_s)
            if year < 100:
                year += 2000
        try:
            day = date(year, int(m_s), int(d_s))
        except ValueError:
            return empty
        if not y_s and day < today:
            try:
                day = day.replace(year=year + 1)
            except ValueError:
                return empty
        return {**empty, **_zip(_at(day, moment))}

    # --- Espressioni a grana di settimana o mese ---
    if re.search(r"\bfine settimana\b|\bweekend\b", text):
        ahead = (5 - today.weekday()) % 7 or 7   # il prossimo sabato
        return {**empty, **_zip(_at(today + timedelta(days=ahead), moment))}
    if re.search(r"\b(settimana prossima|prossima settimana)\b", text):
        ahead = 7 - today.weekday()
        return {"at": datetime.combine(today + timedelta(days=ahead), moment or time(0, 0)).isoformat(),
                "precision": PRECISION_TIME if moment else PRECISION_WEEK,
                "phrase": empty["phrase"]}
    if re.search(r"\bfine mese\b|\bentro il mese\b", text):
        last = _add_months(today.replace(day=1), 1) - timedelta(days=1)
        return {**empty, **_zip(_at(last, moment))}
    if re.search(r"\bmese prossimo\b|\bprossimo mese\b", text):
        day = _add_months(today.replace(day=1), 1)
        return {"at": datetime.combine(day, moment or time(0, 0)).isoformat(),
                "precision": PRECISION_TIME if moment else PRECISION_WEEK,
                "phrase": empty["phrase"]}

    # Solo un'ora, senza giorno: si intende oggi.
    if moment is not None and hour_match:
        return {**empty, **_zip(_at(today, moment))}

    return empty


def _zip(pair: tuple[datetime, str]) -> dict[str, str]:
    moment, precision = pair
    return {"at": moment.isoformat(), "precision": precision}


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year = day.year + month // 12
    month = month % 12 + 1
    # Il 31 gennaio + 1 mese non esiste: si scala all'ultimo giorno valido.
    for candidate in range(day.day, 0, -1):
        try:
            return date(year, month, candidate)
        except ValueError:
            continue
    return day


def describe(iso: str, precision: str) -> str:
    """Etichetta leggibile per l'interfaccia, senza inventare precisione."""
    if not iso:
        return "Non specificata"
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return "Non specificata"
    if precision == PRECISION_TIME:
        return moment.strftime("%d/%m/%Y alle %H:%M")
    if precision == PRECISION_WEEK:
        return "settimana del " + moment.strftime("%d/%m/%Y")
    return moment.strftime("%d/%m/%Y")
