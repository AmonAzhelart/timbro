# Timbro

**Dalla voce al verbale.**

> **tìm·bro** *s.m.*
> 1. La qualità che rende una voce riconoscibile fra tutte le altre.
> 2. Il sigillo che rende un documento ufficiale.

Trascrizione di riunioni con riconoscimento degli interlocutori e generazione automatica del verbale — **interamente in locale, a costo di gestione zero**.

Nessuna API a pagamento, nessun servizio cloud, nessun dato che esce dalla macchina. L'unica credenziale richiesta è un token Hugging Face gratuito, necessario solo per scaricare i modelli di diarizzazione.

```
audio/video  →  WhisperX  →  trascrizione + etichette voce  →  Ollama  →  verbale strutturato
```

## Cosa produce

- **Sintesi** dell'incontro
- **Action point** con responsabile, scadenza e priorità
- **Decisioni prese**, distinte dalle semplici proposte
- **Argomenti trattati** e **questioni aperte**
- **Trascrizione integrale** con timestamp e nome di chi parla
- Export in **Markdown** con un click

---

## Requisiti

| | |
|---|---|
| Docker Desktop | con **WSL2** su Windows |
| GPU NVIDIA | driver aggiornati + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| VRAM | 8 GB sufficienti con la configurazione di default |
| Disco | ~15 GB (immagini Docker + modelli) |

Verifica che Docker veda la GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Installazione

### 1. Token Hugging Face

1. Crea un account su [huggingface.co](https://huggingface.co) e genera un token di tipo **Read**: <https://huggingface.co/settings/tokens>
2. **Accetta le condizioni d'uso** di questi due modelli (passaggio obbligatorio, altrimenti la diarizzazione gira ma restituisce etichette vuote):
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>
   - <https://huggingface.co/pyannote/segmentation-3.0>

### 2. Configurazione

```bash
cp .env.example .env      # Windows PowerShell: copy .env.example .env
```

Puoi lasciare `HF_TOKEN` vuoto: si imposta dall'interfaccia. Gli altri parametri hanno default sensati.

### 3. Avvio

```bash
docker compose up -d --build
```

Il primo avvio richiede tempo: scarica le immagini, il modello LLM (~5 GB) e, al primo job, i modelli Whisper e pyannote (~3 GB). Da lì in poi tutto è in cache nei volumi Docker.

Apri **<http://localhost:8080>**, entra in **⚙ Impostazioni**, incolla il token Hugging Face e premi **Verifica**: ti dice subito se il token è valido e se hai accettato le condizioni dei repository.

L'indicatore in alto mostra lo stato: GPU rilevata, modello Whisper attivo, Ollama raggiungibile, token HF presente. Se qualcosa è rosso, il problema è lì.

---

## Uso

1. Trascina il file audio (o video) nel riquadro di upload. **Non parte subito**: il file resta nel browser finché non confermi.
2. Compila **Dettagli di questa riunione** — di cosa si parla, i nomi che ricorrono, quante persone hanno partecipato. Sono i tre dati che cambiano di più il risultato. Lasciando vuoto l'argomento, viene dedotto automaticamente.
3. **▶ Avvia elaborazione.** Da lì in poi lo stepper mostra la fase corrente con la percentuale reale, e puoi **annullare in qualsiasi momento**: l'annullamento è cooperativo, si applica al primo punto utile — durante le fasi lunghe arriva a ogni spezzone di audio, quindi in pratica entro pochi secondi.
4. Nella scheda **Partecipanti** assegni un nome a `SPEAKER_00`, `SPEAKER_01`, ecc.
5. **Salva e rigenera verbale**: l'LLM riscrive il verbale usando i nomi reali, senza ri-trascrivere l'audio (operazione veloce).
6. Scarica il `.md`.

Se il risultato non convince, la scheda **Contesto** mostra cosa il sistema ha capito della riunione e permette di correggerlo e ri-trascrivere.

Formati accettati: `mp3`, `wav`, `m4a`, `flac`, `ogg`, `opus`, `aac`, `wma`, `mp4`, `mkv`, `mov`, `avi`, `webm`.

---

## Configurazione

Tutto è regolabile da **⚙ Impostazioni** nell'interfaccia, senza toccare file né riavviare i container. Le modifiche vengono salvate in `data/settings.json` sulla tua macchina e si applicano dal job successivo.

### Precedenza dei valori

```
default nel codice   <   .env   <   ⚙ Impostazioni (data/settings.json)
```

`settings.json` contiene **solo i campi che hai effettivamente cambiato**: tutto il resto continua a seguire `.env`. Se riporti un campo al valore di `.env`, l'override sparisce da solo. In fondo alla modale trovi quante impostazioni stai sovrascrivendo.

Il token Hugging Face è salvato con permessi `600` e non viene mai restituito in chiaro dall'API: l'interfaccia ne mostra solo un'anteprima mascherata (`hf_…mnop`).

### Cosa puoi configurare dall'interfaccia

**Credenziali** — token Hugging Face, con pulsante *Verifica* che controlla la validità del token **e** l'accesso ai repository gated. È il controllo che ti evita il problema più insidioso: diarizzazione che gira senza errori ma restituisce etichette vuote.

**Trascrizione** — modello Whisper (con VRAM richiesta indicata per ciascuno), precisione di calcolo, lingua, batch size, modello di diarizzazione, beam size e soglie. Sono i parametri tecnici, quelli che valgono per tutte le riunioni: argomento, termini e partecipanti si impostano invece sul singolo file al caricamento.

**Modello linguistico** — scelta del modello Ollama, elenco dei modelli installati con dimensione su disco, **download di nuovi modelli con barra di avanzamento** e rimozione di quelli inutilizzati. Più finestra di contesto, timeout e soglia di chunking.

**Generale** — dimensione massima degli upload, cancellazione automatica dell'audio dopo l'elaborazione.

Il salvataggio è bloccato mentre un job è in corso, per non cambiare i modelli sotto i piedi al worker.

### Configurazione consigliata per 24 GB (RTX 4090)

| | Valore | VRAM |
|---|---|---|
| Modello Whisper | `large-v3` | ~10 GB |
| Precisione | `float16` | |
| Batch size | `24` | |
| Ampiezza di ricerca | `8` | |
| Modello LLM | **`gemma4:26b`** (MoE, 4B attivi) | ~18 GB |
| Permanenza in VRAM | `Scarica subito` | |

`gemma4:26b` è un MoE: pesa 18 GB ma ne attiva solo 4B per token, quindi va circa alla velocità di un modello molto più piccolo. `gemma4:31b` è denso — qualità leggermente superiore, sensibilmente più lento. Serve Ollama 0.22+.

Whisper e Ollama non devono coesistere: la pipeline libera la VRAM prima di passare all'LLM. Ma non vale il contrario — su job consecutivi il modello linguistico è ancora caricato quando Whisper riparte, e con 18-20 GB su 24 questo basta a mandare in out-of-memory il job dopo. In **⚙ Impostazioni → Permanenza in VRAM** scegli *Scarica subito*: paghi ~10 secondi di ricaricamento per verbale.

Con 24 GB puoi anche attivare la **separazione delle voci** senza scendere a compromessi: gira dopo la diarizzazione, quando Whisper è già stato scaricato.

### Modello di trascrizione

| Modello | VRAM | Velocità | Qualità |
|---|---|---|---|
| `medium` | ~5 GB | veloce | buona |
| `large-v3-turbo` | ~6 GB | molto veloce | cede su audio difficile |
| **`large-v3`** (default) | ~10 GB | normale | massima |

`large-v3-turbo` ha un decoder distillato: su audio pulito è quasi indistinguibile, ma su registrazioni ambientali, accenti marcati o frasi che mescolano italiano e inglese sbaglia sensibilmente di più. Se la trascrizione non ti convince, questo è il primo interruttore da girare.

Se vai in out-of-memory: abbassa il **batch size** (16 → 8 → 4) oppure passa a precisione `int8_float16`.

---

## Migliorare la qualità della trascrizione

Whisper decodifica ogni spezzone di audio **in modo indipendente**: non ha memoria di ciò che è stato detto prima e non sa di cosa si sta parlando. È il motivo per cui sbaglia proprio dove conta — nomi di persone, sigle, prodotti, termini inglesi in mezzo all'italiano.

### Dove si impostano le cose

La distinzione è netta e vale la pena tenerla a mente:

| | Dove | Esempi |
|---|---|---|
| **Cambia a ogni riunione** | Al caricamento del file, sotto *Dettagli di questa riunione* | argomento, nomi e termini ricorrenti, numero di partecipanti |
| **Vale per tutte** | ⚙ Impostazioni | modello Whisper, precisione, beam size, soglie, LLM, token |

Argomento e partecipanti sono proprietà **della singola riunione**: metterli nella configurazione globale significherebbe riscriverli ogni volta e dimenticarsene la volta dopo. Restano salvati sul job, quindi li ritrovi nella scheda **Contesto** e puoi correggerli e ri-trascrivere.

### Ricognizione automatica (attiva di default)

Il sistema si costruisce il contesto da solo, in due passaggi:

1. **Ricognizione** — campiona 4-5 spezzoni da 2 minuti distribuiti sulla riunione (mai più di 10 minuti in tutto, anche su un file da 2 ore) e li trascrive con un modello leggero. Il testo che esce è scadente: serve solo a inquadrare l'argomento.
2. **Deduzione** — l'LLM legge quel testo grezzo e ne ricava argomento, ambito e vocabolario.
3. **Trascrizione vera** — parte con quel contesto già impostato.

Costa 15-30 secondi in più per job. Quello che ha dedotto lo trovi nella scheda **Contesto** del job.

**Come evita di peggiorare le cose.** Un termine sbagliato è peggio di nessun termine: Whisper lo privilegia, quindi trasforma un errore occasionale in un errore sistematico. Per questo l'LLM deve **ancorare** ogni termine al testo da cui lo ricava, e l'ancora viene verificata contro la trascrizione di ricognizione. Se dichiara di aver sentito qualcosa che non c'è, il termine viene scartato.

Non è un filtro basato sulla somiglianza, ed è una scelta deliberata: su un caso reale il recupero corretto (`environment` da "envenot found") e l'invenzione (`Salesforce` da "salvatore") stanno a quattro centesimi di distanza. Qualunque soglia che tenga il primo lascia passare il secondo.

L'ancoraggio elimina l'errore frequente — aggiungere tecnologie plausibili mai nominate — ma un abbinamento sbagliato fra ancora reale e termine errato resta possibile. Per questo la scheda **Contesto** mostra per ogni termine il testo grezzo da cui viene: un "Salesforce ← salvatore" salta all'occhio, lo togli e ri-trascrivi con un clic.

Il contesto scritto a mano nelle impostazioni **non viene sostituito**: quello dedotto si aggiunge in coda.

### Regolazioni manuali

Le leve sono in **⚙ Impostazioni → Qualità della trascrizione**, in ordine di efficacia:

**1. Modello `large-v3`** invece di turbo, se hai la VRAM.

**2. Pulizia audio.** Passa-alto a 80 Hz, riduzione rumore e livellamento dei volumi prima della trascrizione. Su una registrazione ambientale con più persone a distanze diverse dal microfono recupera parecchie parole; su audio già pulito è solo tempo in più.

**3. Beam size** da 5 a 8: più accurato, più lento.

**4. Soglia di silenzio** a 0.4 se perde frasi dette a voce bassa; alzala se il modello inventa testo durante i silenzi.

**5. Audio multilingue** se si passa davvero e spesso da una lingua all'altra: rileva la lingua spezzone per spezzone. Se l'audio è italiano con qualche termine inglese, lascialo spento e usa invece contesto e termini chiave.

Le impostazioni tecniche valgono dal job successivo. Su un job già fatto, la scheda **Contesto** ti permette di correggere argomento e termini e rilanciare la trascrizione senza ricaricare il file.

---

## Quando parlano in due contemporaneamente

È il caso peggiore, e i problemi sono due, distinti.

**Whisper sente un solo flusso.** È addestrato su audio a una voce: della sovrapposizione trascrive quella più forte e l'altra la perde, oppure mescola le due in una frase priva di senso.

**WhisperX butta via l'informazione.** `assign_word_speakers` assegna a ogni parola lo speaker con la sovrapposizione temporale maggiore, quindi il fatto che in quel punto fossero in due sparisce senza lasciare traccia.

### Cosa fa il sistema, senza modelli aggiuntivi

**Spezza i segmenti sul cambio di voce.** WhisperX etichetta ogni singola parola, ma poi assegna al segmento un solo parlante: il dominante. In uno scambio serrato — *«hai finito il report?» «sì, ieri sera»* — l'intero scambio finiva attribuito a una persona sola. Ora le etichette di parola vengono usate per dividere il segmento nei turni reali. È il miglioramento che si nota di più, e non costa nulla.

**Rileva le zone di sovrapposizione** con una sweep line sugli intervalli di diarizzazione, scartando gli accavallamenti sotto 0.4 s che sono rumore di confine fra due turni, non due persone che parlano insieme.

**Emette una riga per ciascuna voce presente**, con il parlante dominante per primo. Il testo resta quello che Whisper è riuscito a sentire, ma almeno si sa chi c'era. Le righe sono marcate nella trascrizione e nel Markdown.

**Istruisce l'LLM a non fidarsi** di quei passaggi: niente decisioni o action point attribuiti a qualcuno sulla base di parlato sovrapposto. Al massimo finiscono fra le questioni aperte, senza responsabile.

### Separazione delle voci (opzionale)

Il fatto che Whisper senta una voce sola si risolve solo separando l'audio in tracce distinte, una per persona, e trascrivendole separatamente. Usa [`pyannote/speech-separation-ami-1.0`](https://huggingface.co/pyannote/speech-separation-ami-1.0), addestrato proprio su riunioni registrate con microfono singolo a distanza.

**Vive in un container a parte**, e non per capriccio: quel pipeline richiede pyannote.audio 3.3.x, mentre WhisperX 3.8.6 richiede la 4.x. Nello stesso ambiente non convivono. Isolarlo permette di pinnare la versione giusta senza toccare la pipeline principale, che continua a funzionare identica quando la separazione è spenta.

```bash
# Accetta le condizioni d'uso di ENTRAMBI i repository:
#   https://huggingface.co/pyannote/speech-separation-ami-1.0
#   https://huggingface.co/pyannote/separation-ami-1.0

docker compose --profile separation up -d --build
```

La prima build richiede diversi minuti: `pesq`, dipendenza transitiva di asteroid, non ha wheel precompilate e viene compilata da sorgente. Il toolchain viene installato e rimosso nello stesso layer, quindi non finisce nell'immagine finale.

Le dipendenze di questo servizio hanno **limiti superiori espliciti**, e non è pignoleria: pyannote.audio 3.3.2 è di fine 2024 e dichiara solo vincoli `>=`, quindi senza tetti pip risolve alle versioni odierne, avanti di anni. Il caso peggiore è `huggingface_hub`, la cui serie 1.x ha rimosso `use_auth_token` — cioè esattamente il parametro con cui questo pipeline carica il modello. Verrebbe fuori un container che builda senza errori e poi esplode al primo caricamento.

Poi in **⚙ Impostazioni → Diagnostica separazione** verifichi che carichi davvero, prima di lanciare un job da un'ora. L'opzione si attiva **per singola riunione**, al caricamento del file: se il servizio non è attivo la casella è disabilitata, non fallisce a metà job.

Come funziona: separa le tracce, trascrive ciascuna per conto suo, poi **sostituisce solo i tratti sovrapposti**. Dove parlava una persona sola la trascrizione della miscela è migliore — la separazione introduce artefatti che lì non servono a niente.

Costi reali: circa 2-4 GB di VRAM in più, tempi sensibilmente più lunghi (separazione + una trascrizione per voce), ~3 GB di immagine Docker aggiuntiva. Sotto l'1% di parlato sovrapposto il sistema salta la separazione da solo e te lo dice: il risultato sarebbe pressoché identico.

### Il rimedio che batte tutti

Se puoi intervenire sulla registrazione, un microfono per persona risolve il problema alla radice e rende superfluo tutto il resto. Nessun modello di separazione eguaglia due tracce già pulite in partenza.

### Indicatore VRAM

In alto trovi l'occupazione della scheda, con la barra divisa fra **trascrizione** (blu) e **LLM** (viola). Si aggiorna ogni 10 secondi, ogni 2 durante un'elaborazione.

Il conteggio viene da tre fonti, perché una sola non basta: Whisper e Ollama girano in **container diversi**, e interrogando CUDA dall'interno del container `api` le allocazioni dell'altro spesso non si vedono — su Docker Desktop/WSL2 in particolare il totale resta inchiodato al valore di riposo anche mentre l'LLM lavora. Quindi:

- **NVML** (driver NVIDIA) per l'occupazione dell'intera scheda,
- **torch** per la memoria riservata da Whisper e pyannote,
- **`/api/ps` di Ollama** per la VRAM del modello caricato — arriva via HTTP, quindi è immune al problema.

Il valore mostrato è il massimo fra quanto riporta il dispositivo e la somma delle quote note: se il conteggio di scheda sottostima, il totale resta comunque veritiero. Passando il mouse vedi la ripartizione e quale fonte è stata usata.

### Modello LLM

| `OLLAMA_MODEL` | VRAM | Note |
|---|---|---|
| **`qwen3:8b`** (default) | ~6 GB | ottimo compromesso, buon italiano |
| `qwen3:14b` | ~10 GB | action point più accurati |
| `gemma3:12b` | ~9 GB | molto solido sul multilingua |
| `llama3.1:8b` | ~5 GB | alternativa leggera |

Per cambiare modello: **⚙ Impostazioni → Scarica**, poi selezionalo come *Modello in uso* e salva. Nessun riavvio, nessun comando.

### Numero di interlocutori

Si indica al caricamento del file, sotto *Dettagli di questa riunione*: se sai quante persone hanno parlato, dirlo migliora sensibilmente il riconoscimento delle voci. Lasciando vuoto, il numero viene stimato automaticamente.

---

## Architettura

```
Timbro/
├── docker-compose.yml        api + ollama + separator (profilo opzionale)
├── separator/                servizio isolato di separazione voci
│   ├── Dockerfile            pyannote 3.3.2 pinnato: incompatibile con la 4.x
│   └── app.py                API HTTP, carica il modello solo quando serve
├── .env                      configurazione iniziale (non versionata)
├── data/                     audio, database, impostazioni, risultati — visibili dall'host
│   ├── timbro.sqlite3
│   └── settings.json         override salvati da ⚙ Impostazioni (permessi 600)
├── backend/
│   ├── Dockerfile            base pytorch/cuda 12.4 + ffmpeg
│   ├── tools/preflight.py    controlli statici eseguiti durante la build
│   └── app/
│       ├── main.py           API FastAPI + serving della UI
│       ├── config.py         settings da env
│       ├── settings_store.py override persistenti, validazione, mascheramento token
│       ├── models.py         schemi Pydantic
│       ├── store.py          persistenza SQLite
│       ├── pipeline.py       worker in background, 1 job GPU alla volta
│       ├── transcribe.py     WhisperX: ricognizione + ASR + allineamento + diarizzazione
│       ├── recon.py          campionamento audio e filtro anti-invenzione dei termini
│       ├── overlap.py        segmenti spezzati sul cambio voce, zone sovrapposte
│       ├── separation.py     client verso il servizio di separazione
│       ├── summarize.py      Ollama con output JSON vincolato da schema
│       ├── ollama_admin.py   elenco / download / rimozione modelli LLM
│       ├── hf.py             verifica token e accesso ai repo gated
│       ├── gpu.py            lettura occupazione VRAM
│       ├── prompts.py        prompt in italiano
│       └── export.py         generazione Markdown
└── frontend/index.html       UI single-file, zero dipendenze
```

**Scelte che tengono la manutenzione a zero:**

- **SQLite** invece di Postgres: un file in `data/`, nessun servizio da presidiare.
- **Worker in-process** invece di Celery/Redis: una coda thread-safe, un job GPU alla volta.
- **Modelli in cache su volumi Docker**: scaricati una sola volta, sopravvivono ai rebuild.
- **VRAM liberata esplicitamente** fra la fase Whisper e la fase LLM, così i due modelli non si contendono la GPU.
- **Cache modelli invalidata da sola** quando cambi modello o token dalle impostazioni: il ricaricamento avviene al job successivo, senza riavviare nulla.
- **Job interrotti da un riavvio** vengono marcati in errore all'avvio e sono rilanciabili con un click.
- **Frontend senza build step**: un file HTML, niente npm, niente da aggiornare.

### Riunioni lunghe

Oltre `CHUNK_CHARS` caratteri la trascrizione viene analizzata in **map-reduce**: appunti per singolo blocco, poi sintesi finale. Una riunione di 3 ore viene gestita senza saturare il contesto del modello.

---

## API

La UI usa queste rotte; sono utilizzabili anche da script. Documentazione interattiva su <http://localhost:8080/docs>.

| Metodo | Rotta | Descrizione |
|---|---|---|
| `POST` | `/api/jobs` | upload audio (multipart, campo `file`) |
| `GET` | `/api/jobs` | elenco job |
| `GET` | `/api/jobs/{id}` | dettaglio: segmenti + analisi |
| `POST` | `/api/jobs/{id}/speakers` | rinomina voci ed eventuale rigenerazione |
| `POST` | `/api/jobs/{id}/rediarize` | rifà solo il riconoscimento voci |
| `POST` | `/api/jobs/{id}/retranscribe` | ri-trascrive con contesto corretto a mano |
| `POST` | `/api/settings/diagnose-diarization` | carica la pipeline e riporta l'errore esatto |
| `POST` | `/api/settings/diagnose-separation` | verifica il servizio di separazione voci |
| `GET` | `/api/jobs/{id}/markdown` | verbale in Markdown (`?download=true`) |
| `POST` | `/api/jobs/{id}/cancel` | annulla un'elaborazione in corso o in coda |
| `POST` | `/api/jobs/{id}/retry` | rilancia un job fallito |
| `DELETE` | `/api/jobs/{id}` | elimina job e audio |
| `GET` | `/api/health` | diagnostica GPU / Ollama / token |
| `GET` | `/api/gpu` | occupazione VRAM in tempo reale |
| `GET` | `/api/settings` | impostazioni correnti (token mascherato) |
| `PUT` | `/api/settings` | aggiornamento parziale, salvato su disco |
| `GET` | `/api/settings/options` | valori proposti nelle tendine |
| `POST` | `/api/settings/verify-hf` | verifica token e accesso ai repo gated |
| `GET` | `/api/settings/llms` | modelli Ollama installati |
| `POST` | `/api/settings/llms/pull` | avvia il download di un modello |
| `GET` | `/api/settings/llms/pull` | avanzamento del download |
| `DELETE` | `/api/settings/llms/{name}` | rimuove un modello |

Esempio da riga di comando:

```bash
curl -F "file=@riunione.mp3" http://localhost:8080/api/jobs
curl http://localhost:8080/api/jobs/<id>/markdown > verbale.md
```

---

## Problemi frequenti

**La trascrizione non distingue chi parla (tutti `SPEAKER_?`)**
Apri **⚙ Impostazioni → Diagnostica** accanto al modello di diarizzazione: carica davvero la pipeline e ti dice l'errore esatto, distinguendo token invalido, licenza non accettata e incompatibilità di libreria. Sistemato il problema, usa **↻ Rifai riconoscimento voci** sul job: la trascrizione non viene rifatta, solo l'etichettatura e il verbale.

Il caso più frequente è la licenza: le condizioni d'uso vanno accettate con lo **stesso account** che ha generato il token, e su `speaker-diarization-community-1` (non solo sul vecchio `3.1`).

**Rileva una sola voce quando i partecipanti erano di più**
Apri la scheda **Contesto** del job, indica minimo e massimo partecipanti e ri-trascrivi; oppure usa **↻ Rifai riconoscimento voci**, che riusa i valori salvati sul job. Su audio con voci simili o molto sovrapposte il rilevamento automatico tende a sottostimare.

**`CUDA out of memory`**
Da ⚙ Impostazioni: abbassa il batch size, passa a `medium` o `large-v3-turbo`, oppure scegli un LLM più piccolo. Whisper e Ollama condividono la stessa GPU: il codice libera la VRAM fra le due fasi, ma con meno di 8 GB conviene ridimensionare entrambi.

**Ollama offline nell'indicatore di stato**
`docker compose logs ollama`. Se manca il modello, scaricalo da ⚙ Impostazioni → *Scarica*, oppure da terminale: `docker compose exec ollama ollama pull qwen3:8b`.

**Non riesco a salvare le impostazioni**
Il salvataggio è bloccato durante un'elaborazione. Aspetta che il job finisca.

**Voglio ripartire dalla configurazione di `.env`**
Cancella `data/settings.json` e riavvia: `docker compose restart api`.

**Quando si sovrappongono, attribuisce male le battute**
Vedi [Quando parlano in due contemporaneamente](#quando-parlano-in-due-contemporaneamente). La trascrizione marca i tratti dubbi; per recuperare il parlato coperto serve la separazione delle voci, che si attiva al caricamento del file.

**«Servizio non attivo» sulla separazione voci**
Il container `separator` non è avviato: `docker compose --profile separation up -d`. Se è avviato ma la diagnostica fallisce, mancano le condizioni d'uso su Hugging Face — servono su **entrambi** i repository, `speech-separation-ami-1.0` e `separation-ami-1.0`.

**Sbaglia nomi propri, sigle e termini inglesi**
Normale senza contesto: vedi [Migliorare la qualità della trascrizione](#migliorare-la-qualità-della-trascrizione). Al caricamento apri *Dettagli di questa riunione* e compila argomento e termini: sono due minuti e cambiano molto il risultato.

**Trascrizione lentissima**
Stai girando su CPU. Verifica `/api/health` → `device`: deve dire `cuda`. In caso contrario manca il NVIDIA Container Toolkit.

**`docker compose up` fallisce con "env file .env not found"**
Non hai copiato `.env.example` in `.env`.

---

## Controlli statici

`backend/tools/preflight.py` replica gli assert che FastAPI e Pydantic eseguono all'avvio, senza bisogno di installare nulla: rotte con status incompatibile, parametri di path mancanti, riferimenti a funzioni inesistenti fra moduli, campi Pydantic che collidono con `BaseModel`, chiamate della UI verso endpoint inesistenti.

Gira automaticamente durante `docker compose build`: se qualcosa è malformato la build fallisce subito, invece di lasciarti un container in crash-loop. Puoi lanciarlo anche a mano, richiede mezzo secondo:

```bash
python3 backend/tools/preflight.py
```

---

## Manutenzione

Praticamente nulla. Le uniche operazioni periodiche opzionali:

```bash
docker compose pull && docker compose up -d --build   # aggiornare le immagini
docker compose logs -f api                            # seguire i log
docker compose down                                   # fermare tutto (i dati restano)
```

I file in `data/` sono tuoi: audio originali, database e risultati restano sul disco anche a container spenti.

---

Sources: [WhisperX](https://github.com/m-bain/whisperX) · [pyannote.audio](https://huggingface.co/pyannote/speaker-diarization-community-1) · [Ollama](https://ollama.com)
#   t i m b r o 
 
 