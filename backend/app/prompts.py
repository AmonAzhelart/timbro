"""Prompt in italiano per la ricognizione e per l'analisi della riunione."""

# --- Ricognizione: deduce il contesto da una trascrizione grezza ------------
RECON_SYSTEM = (
    "Sei un assistente che prepara il vocabolario per un sistema di trascrizione "
    "automatica. Ricevi una trascrizione GREZZA e piena di errori, prodotta da un "
    "modello veloce senza contesto. Il tuo compito non è correggerla, ma capire di "
    "che cosa si parla, così che una seconda trascrizione più accurata parta "
    "avvantaggiata. Rispondi solo con JSON valido, senza testo attorno."
)

RECON = """Questa è una trascrizione grezza e imprecisa di alcuni estratti di una riunione.
Contiene sicuramente parole storpiate: è normale ed è proprio il motivo per cui esisti.

--- ESTRATTI ---
{draft}
--- FINE ESTRATTI ---

Deduci il contesto e restituisci un oggetto JSON con questa struttura:

{{
  "topic": "una o due frasi che descrivono di cosa si parla e in che lingua, \
scritte come istruzione per un trascrittore (es. \\"Riunione tecnica in italiano \
con termini inglesi su sviluppo software, container e API.\\")",
  "domain": "ambito in due o tre parole (es. sviluppo software, vendite, sanità)",
  "languages": ["it"],
  "terms": [
    {{"term": "la parola corretta", "heard": "come compare NEGLI ESTRATTI QUI SOPRA"}}
  ],
  "confidence": "alta | media | bassa"
}}

Regole vincolanti su "terms" — leggile con attenzione:
- Includi SOLO nomi propri di persone, aziende, prodotti, tecnologie, sigle e \
parole tecniche ricorrenti.
- Il campo "heard" deve contenere una sequenza di parole COPIATA LETTERALMENTE \
dagli estratti, carattere per carattere. Se non riesci a indicare da quale punto \
del testo ricavi il termine, allora quel termine NON va incluso.
- Il campo "term" è la forma corretta. Se la parola era già corretta negli \
estratti, "term" e "heard" coincidono.
- Esempio di correzione valida: se negli estratti compare "envenot found", \
allora {{"term": "environment", "heard": "envenot found"}}.
- Esempio di ciò che NON devi fare: aggiungere una tecnologia solo perché è \
plausibile nel contesto. Se non l'hai sentita, non esiste.
- Un termine sbagliato viene RINFORZATO nella trascrizione successiva e peggiora \
il risultato; un termine mancante lo lascia semplicemente com'è. Nel dubbio, ometti.
- Niente parole comuni della lingua ("quindi", "allora", "comunque").
- Massimo 25 termini, ordinati dal più al meno affidabile.
- Se gli estratti sono troppo confusi per capire qualcosa, restituisci "terms": [] \
e "confidence": "bassa".

Regole su "topic":
- Descrivi l'argomento, non riassumere la conversazione.
- Indica sempre la lingua parlata e segnala se si mescolano più lingue.
- Massimo 40 parole.

Rispondi SOLO con il JSON."""


# --- Analisi della riunione -------------------------------------------------

SYSTEM = (
    "Sei un assistente esperto nella redazione di verbali di riunione. "
    "Lavori esclusivamente sul testo della trascrizione che ti viene fornito: "
    "non inventi fatti, nomi, numeri o scadenze. "
    "Se un'informazione non è presente nella trascrizione, lo dichiari esplicitamente. "
    "Rispondi sempre in italiano e restituisci esclusivamente JSON valido, "
    "senza testo introduttivo e senza blocchi di codice markdown."
)

# --- Fase MAP: note per singolo blocco di riunioni lunghe -------------------
MAP = """Questa è la porzione {index} di {total} della trascrizione di una riunione.
Gli interlocutori sono identificati all'inizio di ogni riga.

--- TRASCRIZIONE (porzione {index}/{total}) ---
{chunk}
--- FINE PORZIONE ---

Estrai da QUESTA porzione, in italiano e in forma di appunti sintetici:
1. Argomenti trattati e punti salienti della discussione.
2. Decisioni prese (solo se esplicitamente concordate, non ipotesi).
3. Azioni/impegni assunti, indicando chi se ne occupa e la scadenza se citata.
4. Questioni rimaste aperte o rinviate.

Non ripetere frasi testuali lunghe: riassumi. Se una categoria è assente, scrivi "nessuna".
Rispondi in testo semplice, massimo 400 parole."""

# --- Fase REDUCE / SINGLE-PASS: output strutturato --------------------------
REDUCE = """Di seguito {source_desc} di una riunione.

--- CONTENUTO ---
{content}
--- FINE CONTENUTO ---

Produci il verbale della riunione come oggetto JSON con ESATTAMENTE questa struttura:

{{
  "title": "titolo breve e descrittivo della riunione (max 10 parole)",
  "overview": "sintesi complessiva in 3-5 frasi: scopo dell'incontro ed esito",
  "sections": [
    {{"title": "argomento trattato", "content": "cosa è stato detto, 2-5 frasi"}}
  ],
  "decisions": [
    {{"decision": "decisione presa", "context": "motivazione o vincolo emerso"}}
  ],
  "action_points": [
    {{"task": "azione concreta da svolgere",
      "owner": "nome della persona responsabile oppure 'Non assegnato'",
      "due": "scadenza indicata oppure 'Non specificata'",
      "priority": "alta | media | bassa"}}
  ],
  "open_questions": ["questione rimasta senza risposta"]
}}

Regole vincolanti:
- Usa i NOMI degli interlocutori come compaiono nel contenuto (es. "Marco", "SPEAKER_01").
- In "decisions" inserisci solo scelte effettivamente concordate, non proposte o ipotesi.
- In "action_points" ogni voce deve iniziare con un verbo all'infinito (es. "Preparare il preventivo").
- Non attribuire un owner se non è deducibile: usa "Non assegnato".
- Non inventare date: se la scadenza non è citata usa "Non specificata".
- Se un elenco non ha elementi, restituisci una lista vuota [].
- Rispondi SOLO con il JSON, nient'altro."""
