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
    "Sei un verbalizzante esperto: redigi verbali analitici di riunione, non riassunti. "
    "Il tuo verbale è un documento di registro: chi non era presente deve poter "
    "ricostruire dalla tua sola lettura di cosa si è parlato, quali posizioni sono "
    "state espresse, con quali argomenti e come si è concluso. "
    "Lavori esclusivamente sul testo della trascrizione che ti viene fornito: "
    "non inventi fatti, nomi, numeri o scadenze. "
    "Se un'informazione non è presente nella trascrizione, lo dichiari esplicitamente. "
    "Fra un verbale ridondante e uno incompleto scegli sempre il primo: omettere "
    "un passaggio discusso è l'errore più grave che puoi commettere. "
    "Rispondi sempre in italiano e restituisci esclusivamente JSON valido, "
    "senza testo introduttivo e senza blocchi di codice markdown."
)

# --- Fase MAP: note per singolo blocco di riunioni lunghe -------------------
# NB: questa fase è il collo di bottiglia della qualità. Ciò che non finisce
# negli appunti non esiste più per la fase REDUCE, che non rivede la
# trascrizione. Per questo qui si ricostruisce, non si riassume.
MAP = """Questa è la porzione {index} di {total} della trascrizione di una riunione.
Gli interlocutori sono identificati all'inizio di ogni riga.

--- TRASCRIZIONE (porzione {index}/{total}) ---
{chunk}
--- FINE PORZIONE ---

Redigi appunti ANALITICI e DETTAGLIATI su questa porzione. Non stai riassumendo:
stai ricostruendo la discussione per qualcuno che dovrà scrivere il verbale
ufficiale senza poter rileggere la trascrizione. Tutto ciò che non scrivi qui
andrà perduto.

Struttura gli appunti così:

## Argomenti
Per OGNI argomento affrontato, anche se marginale o durato pochi scambi, un
paragrafo che riporti:
- chi ha introdotto il tema e perché;
- cosa ha detto ciascun interlocutore, nominandolo;
- le posizioni divergenti, le obiezioni, i dubbi e le controproposte;
- ogni dato concreto citato: cifre, importi, percentuali, date, versioni, nomi
  di file, sistemi, aziende, clienti, strumenti, riferimenti normativi;
- gli esempi e i casi concreti portati a sostegno;
- come il tema si è chiuso: accordo, rinvio, o nessuna conclusione.

## Decisioni
Solo scelte effettivamente concordate, con la motivazione emersa e le
alternative scartate. Se non ci sono decisioni, scrivi "nessuna".

## Impegni
Chi ha detto di fare cosa, con la scadenza se citata. Se non ci sono, scrivi
"nessuno".

## Questioni aperte
Domande rimaste senza risposta, punti rinviati, informazioni che mancavano ai
presenti per decidere. Se non ci sono, scrivi "nessuna".

Regole vincolanti:
- NON accorpare argomenti diversi per fare sintesi: tienili distinti.
- NON scrivere formule vuote come "si è discusso di vari aspetti" o "sono stati
  approfonditi alcuni punti": indica sempre QUALI, con i dettagli.
- Riporta fra virgolette la frase testuale quando la formulazione esatta conta
  (un impegno, una cifra, una condizione, un disaccordo netto).
- Conserva i nomi propri e i termini tecnici esattamente come compaiono.
- La lunghezza degli appunti deve essere proporzionale a quanto è stato detto in
  questa porzione: non c'è un limite massimo, l'unico limite è non aggiungere
  nulla che non sia nella trascrizione.

Rispondi in testo semplice."""

# --- Fase REDUCE / SINGLE-PASS: output strutturato --------------------------
REDUCE = """Di seguito {source_desc} di una riunione.

--- CONTENUTO ---
{content}
--- FINE CONTENUTO ---

Produci il verbale analitico della riunione come oggetto JSON con ESATTAMENTE
questa struttura:

{{
  "title": "titolo breve e descrittivo della riunione (max 10 parole)",
  "overview": "sintesi complessiva in 6-10 frasi: scopo dell'incontro, i filoni \
principali della discussione, l'esito e ciò che resta da fare",
  "sections": [
    {{"title": "argomento trattato",
      "content": "resoconto esteso e dettagliato di quell'argomento"}}
  ],
  "decisions": [
    {{"decision": "decisione presa",
      "context": "motivazione, vincoli emersi e alternative scartate"}}
  ],
  "action_points": [
    {{"task": "azione concreta da svolgere",
      "owner": "nome della persona responsabile oppure 'Non assegnato'",
      "due": "la scadenza ESATTAMENTE COME È STATA DETTA, oppure 'Non specificata'",
      "priority": "alta | media | bassa"}}
  ],
  "open_questions": ["questione rimasta senza risposta"]
}}

Come scrivere le "sections" — è la parte più importante del verbale:
- Una sezione per OGNI argomento affrontato, anche se marginale. Non accorpare
  temi distinti e non scartare i minori: se sono stati discussi, entrano.
- Ogni sezione deve ricostruire la discussione, non annunciarla. Includi: chi ha
  sollevato il tema e perché, cosa ha sostenuto ciascun interlocutore, le
  obiezioni e le posizioni divergenti, i dati concreti citati (cifre, date,
  nomi, sistemi, importi, riferimenti), gli esempi portati, e come il punto si è
  chiuso o perché è rimasto aperto.
- Scrivi in prosa distesa, più paragrafi se serve. Nessun limite di lunghezza:
  una sezione su un tema discusso a lungo deve essere lunga.
- Vietate le formule vuote: "sono stati discussi diversi aspetti", "si è parlato
  di varie questioni", "il team ha approfondito il tema". Se scrivi una frase
  del genere, sostituiscila indicando esattamente quali aspetti e cosa è emerso.
- Ordina le sezioni seguendo lo svolgimento della riunione.
- Se il contenuto proviene da appunti di più porzioni, unisci le occorrenze dello
  stesso argomento in un'unica sezione SOMMANDO i dettagli, senza sceglierne uno
  e scartare gli altri.

Regole vincolanti su tutto il resto:
- Usa i NOMI degli interlocutori come compaiono nel contenuto (es. "Marco", "SPEAKER_01").
- In "decisions" inserisci solo scelte effettivamente concordate, non proposte o ipotesi.
- In "action_points" ogni voce deve iniziare con un verbo all'infinito (es. "Preparare il preventivo").
- Non attribuire un owner se non è deducibile: usa "Non assegnato".

Regole sul campo "due" — leggile con attenzione, è cambiato:
- Riporta la scadenza CON LE PAROLE USATE IN RIUNIONE: "entro domattina",
  "venerdì prossimo", "fra due settimane", "il 15 settembre", "entro le 18".
- NON convertire in una data di calendario e NON calcolare nulla: al calcolo
  pensa il programma, che conosce data e ora esatte della riunione. Se provi a
  contare i giorni sbagli, e l'errore diventa una scadenza sbagliata sul
  calendario di qualcuno.
- Una scadenza relativa È una scadenza: "domattina" va riportato, non
  scartato come se non fosse stato detto nulla.
- Usa "Non specificata" SOLO quando davvero non è stato indicato alcun
  termine. "Appena possibile" e "quando sarà pronto" sono termini vaghi ma
  reali: riportali come sono.
- Non introdurre nulla che non sia nel contenuto qui sopra.
- Se un elenco non ha elementi, restituisci una lista vuota [].
- Rispondi SOLO con il JSON, nient'altro."""

# --- Fase FOLD: consolidamento di appunti troppo voluminosi -----------------
# Serve solo alle riunioni molto lunghe: appunti dettagliati su decine di
# porzioni non entrano nel contesto della fase REDUCE. Consolidiamo per gruppi
# invece di lasciare che il contesto venga troncato in silenzio, perdendo
# proprio le parti finali della riunione.
FOLD = """Di seguito gli appunti analitici di porzioni consecutive della stessa riunione.

--- APPUNTI ---
{content}
--- FINE APPUNTI ---

Uniscili in un unico blocco di appunti continuo, mantenendo la stessa struttura
(## Argomenti, ## Decisioni, ## Impegni, ## Questioni aperte).

Regole vincolanti:
- Questo NON è un riassunto: è una fusione. Non devi accorciare, devi ricucire.
- Se lo stesso argomento compare in più porzioni, uniscilo in un unico paragrafo
  sommando i dettagli di entrambe, senza scartarne nessuno.
- Conserva integralmente nomi, cifre, date, termini tecnici e citazioni testuali.
- Non eliminare gli argomenti minori.
- Non aggiungere nulla che non sia negli appunti qui sopra.

Rispondi in testo semplice."""
