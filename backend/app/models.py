"""Schemi Pydantic condivisi fra pipeline, API e frontend."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    SCOUTING = "scouting"
    TRANSCRIBING = "transcribing"
    SEPARATING = "separating"
    ALIGNING = "aligning"
    DIARIZING = "diarizing"
    SUMMARIZING = "summarizing"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


# Etichette leggibili mostrate nella UI
STATUS_LABELS: dict[str, str] = {
    JobStatus.QUEUED: "In coda",
    JobStatus.SCOUTING: "Ricognizione del contenuto",
    JobStatus.TRANSCRIBING: "Trascrizione audio",
    JobStatus.SEPARATING: "Separazione delle voci",
    JobStatus.ALIGNING: "Allineamento parole",
    JobStatus.DIARIZING: "Riconoscimento voci",
    JobStatus.SUMMARIZING: "Analisi con LLM",
    JobStatus.DONE: "Completato",
    JobStatus.ERROR: "Errore",
    JobStatus.CANCELLED: "Annullato",
}


class Word(BaseModel):
    """Una parola con il proprio istante, prodotta dall'allineamento WhisperX.

    Serve a due cose che il segmento da solo non può dare: illuminare il
    parlato parola per parola durante la riproduzione, e mostrare i cambi di
    voce *dentro* un intervento — altrimenti un blocco in cui due persone si
    alternano sembra detto tutto dalla stessa.

    `start`/`end` sono opzionali: l'allineamento non riesce su ogni parola
    (numeri, sigle, parole coperte da rumore) e in quei casi WhisperX non
    emette gli istanti.
    """

    word: str
    start: float | None = None
    end: float | None = None
    #: Presente solo quando la diarizzazione ha etichettato la singola parola.
    speaker: str | None = None


class Segment(BaseModel):
    start: float
    end: float
    speaker: str = "SPEAKER_?"
    text: str
    #: Vuoto se l'allineamento non è stato eseguito: l'interfaccia ricade
    #: sull'evidenziazione per intervento, che è la resa di prima.
    words: list[Word] = Field(default_factory=list)
    #: Voci attive contemporaneamente in questo tratto. Se valorizzato, il
    #: testo è meno affidabile: Whisper trascrive un solo flusso.
    overlap: list[str] = Field(default_factory=list)
    #: True se questo testo proviene da una traccia audio separata, quindi
    #: appartiene davvero a questa voce e non al parlante dominante.
    separated: bool = False
    #: Su un tratto sovrapposto non separato lo stesso testo viene emesso una
    #: volta per voce presente: senza questo flag l'interfaccia non distingue
    #: l'originale dai duplicati e mostra la stessa frase tre volte di fila,
    #: attribuita a tre persone diverse. True = è la copia del parlante
    #: dominante, quella di cui il testo è più probabilmente fedele.
    overlap_primary: bool = True


class OverlapStats(BaseModel):
    count: int = 0
    total_s: float = 0.0
    percent: float = 0.0
    max_speakers: int = 0
    separated: bool = False


class ActionPoint(BaseModel):
    task: str = Field(description="Azione concreta da svolgere")
    owner: str = Field(default="Non assegnato", description="Chi se ne occupa")
    #: La scadenza COME È STATA DETTA («entro domattina»). Resta testuale
    #: perché è la citazione di ciò che è avvenuto in riunione.
    due: str = Field(default="Non specificata", description="Scadenza indicata")
    #: La stessa scadenza risolta in data vera, calcolata da `dates.resolve`
    #: a partire dall'istante della riunione. Vuoto se non risolvibile.
    due_at: str = Field(default="", description="Scadenza in ISO 8601")
    #: "ora" | "giorno" | "settimana" | "" — a che grana è nota la scadenza.
    #: Serve a non mostrare un orario che nessuno ha mai pronunciato.
    due_precision: str = Field(default="", description="Precisione della scadenza")
    priority: str = Field(default="media", description="alta | media | bassa")
    #: Spuntato a mano dal cruscotto. Vive dentro l'analisi e non in una
    #: tabella a parte: così sopravvive alla modifica manuale del verbale,
    #: che riscrive l'intero oggetto.
    done: bool = False


class Decision(BaseModel):
    decision: str = Field(description="Decisione presa")
    context: str = Field(default="", description="Motivazione o contesto")


class SummarySection(BaseModel):
    title: str
    content: str


class MeetingAnalysis(BaseModel):
    """Output strutturato prodotto dall'LLM."""

    title: str = "Riunione"
    overview: str = ""
    sections: list[SummarySection] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_points: list[ActionPoint] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class Folder(BaseModel):
    id: str
    name: str
    parent_id: str | None = None
    color: str = ""
    created_at: str = ""
    #: Riunioni contenute direttamente, non nelle sottocartelle.
    jobs: int = 0


class FolderRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    parent_id: str | None = None
    color: str = ""


class FolderPatch(BaseModel):
    model_config = {"extra": "forbid"}
    name: str | None = Field(default=None, min_length=1, max_length=80)
    color: str | None = None
    #: `move` distingue "non spostare" da "sposta alla radice", che con il
    #: solo `parent_id=None` sarebbero indistinguibili.
    move: bool = False
    parent_id: str | None = None


class Tag(BaseModel):
    id: str
    name: str
    color: str = ""
    jobs: int = 0


class TagRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    color: str = ""


class Glossary(BaseModel):
    id: str = ""
    name: str = Field(min_length=1, max_length=80)
    prompt: str = Field(default="", max_length=2000)
    hotwords: str = Field(default="", max_length=2000)
    folder_id: str | None = None
    created_at: str = ""


class KnownSpeaker(BaseModel):
    name: str
    uses: int = 1
    last_used: str = ""


class RecordedAtRequest(BaseModel):
    """Correzione manuale della data della riunione."""

    recorded_at: str = Field(description="ISO 8601, es. 2026-08-05T14:30")


class JobFiling(BaseModel):
    """Collocazione di una riunione: cartella ed etichette."""

    model_config = {"extra": "forbid"}
    folder_id: str | None = None
    move: bool = False
    tags: list[str] | None = None


class Task(BaseModel):
    """Un impegno preso in riunione, come entità viva.

    Nasce da un action point ma non ci vive dentro: riprogrammarlo o
    commentarlo non deve riscrivere il verbale, che è il registro di ciò che
    è stato detto e non va falsificato a posteriori.
    """

    id: str = ""
    title: str = Field(min_length=1, max_length=400)
    owner: str = ""
    #: da_fare | in_corso | bloccato | fatto
    status: str = "da_fare"
    priority: str = "media"
    due_at: str = ""
    due_precision: str = ""
    #: La scadenza come fu detta in riunione, conservata anche dopo uno
    #: spostamento: dice da dove si era partiti.
    due_phrase: str = ""
    notes: str = ""
    #: Riunione di origine e punto esatto in cui l'impegno è stato preso.
    job_id: str | None = None
    job_title: str = ""
    ap_index: int | None = None
    source_s: float | None = None
    comments: int = 0
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=400)
    owner: str = ""
    status: str = "da_fare"
    priority: str = "media"
    due_at: str = ""
    due_precision: str = ""
    notes: str = ""
    job_id: str | None = None


class TaskPatch(BaseModel):
    model_config = {"extra": "forbid"}
    title: str | None = Field(default=None, min_length=1, max_length=400)
    owner: str | None = None
    status: str | None = None
    priority: str | None = None
    due_at: str | None = None
    due_precision: str | None = None
    notes: str | None = None
    #: Motivo dello spostamento: finisce nello storico come commento di
    #: sistema. Fra un mese "perché era slittato" vale quanto la data nuova.
    reason: str | None = None


class TaskComment(BaseModel):
    id: str = ""
    body: str = Field(min_length=1, max_length=4000)
    #: "commento" scritto da una persona · "sistema" generato da un'azione
    kind: str = "commento"
    created_at: str = ""


class SearchHit(BaseModel):
    job_id: str
    kind: str
    kind_label: str = ""
    ref: str = ""
    label: str = ""
    snippet: str = ""
    score: float = 0.0
    exact: float = 0.0
    semantic: float = 0.0
    #: Riempiti dal chiamante: la ricerca lavora sull'indice, non sui job.
    job_title: str = ""
    job_created: str = ""
    folder_id: str | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit] = Field(default_factory=list)
    #: True se la parte per significato ha davvero contribuito.
    semantic: bool = False
    note: str | None = None


class OpenActionPoint(ActionPoint):
    """Un impegno con l'indicazione della riunione da cui proviene."""

    job_id: str
    job_title: str = ""
    job_created: str = ""
    index: int = 0


class JobSummary(BaseModel):
    """Vista compatta di un job, usata nella lista."""

    id: str
    filename: str
    status: JobStatus
    status_label: str = ""
    progress: int = 0
    message: str = ""
    error: str | None = None
    created_at: str
    updated_at: str
    duration_s: float | None = None
    #: Quando la riunione è AVVENUTA, non quando è stata caricata. Senza
    #: questo, «entro domattina» non è risolvibile.
    recorded_at: str = ""
    speakers: list[str] = Field(default_factory=list)
    title: str | None = None
    folder_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    #: Problemi non bloccanti (es. diarizzazione fallita). Mai silenziosi.
    warnings: list[str] = Field(default_factory=list)


class JobOptions(BaseModel):
    """Parametri che appartengono alla singola riunione, non alla configurazione.

    Argomento, vocabolario e numero di partecipanti cambiano da un incontro
    all'altro: vivono sul job e si impostano al caricamento del file.
    """

    initial_prompt: str = Field(default="", max_length=2000)
    hotwords: str = Field(default="", max_length=2000)
    min_speakers: int | None = Field(default=None, ge=1, le=30)
    max_speakers: int | None = Field(default=None, ge=1, le=30)
    #: Disattiva la ricognizione per questo job (es. contesto già scritto a mano).
    auto_context: bool = True
    #: Separa le voci sovrapposte in tracce distinte, trascritte una per una.
    #: Richiede il servizio `separator` avviato.
    separate_overlaps: bool = False

    def normalized(self) -> "JobOptions":
        lo, hi = self.min_speakers, self.max_speakers
        if lo and hi and lo > hi:
            lo, hi = hi, lo
        return JobOptions(
            initial_prompt=(self.initial_prompt or "").strip(),
            hotwords=(self.hotwords or "").strip(),
            min_speakers=lo,
            max_speakers=hi,
            auto_context=self.auto_context,
            separate_overlaps=self.separate_overlaps,
        )


class ReconTerm(BaseModel):
    """Termine dedotto, con il testo grezzo da cui è stato ricavato.

    L'ancora è mostrata in revisione: è il modo per accorgersi di un
    abbinamento sbagliato senza dover rileggere tutta la trascrizione.
    """

    term: str
    heard: str = ""
    corrected: bool = False


class ReconReport(BaseModel):
    """Esito della ricognizione: cosa è stato dedotto e cosa è stato applicato."""

    enabled: bool = False
    ok: bool = False
    reason: str | None = None
    sampled_s: int = 0
    topic: str = ""
    domain: str = ""
    confidence: str = "bassa"
    terms: list[ReconTerm] = Field(default_factory=list)
    terms_discarded: int = 0
    draft_excerpt: str = ""
    applied_prompt: str = ""
    applied_hotwords: str = ""
    error: str | None = None


class JobDetail(JobSummary):
    segments: list[Segment] = Field(default_factory=list)
    analysis: MeetingAnalysis | None = None
    speaker_names: dict[str, str] = Field(default_factory=dict)
    recon: ReconReport | None = None
    options: JobOptions = Field(default_factory=JobOptions)
    overlap: OverlapStats | None = None


class RetranscribeRequest(JobOptions):
    """Ri-trascrizione con i parametri corretti a mano.

    Eredita da `JobOptions`: sono gli stessi campi del caricamento, perché
    correggere un job equivale a ricaricarlo con parametri diversi.
    """

    #: Se true la ricognizione viene saltata: comanda quello che scrive l'utente.
    skip_recon: bool = True


class RenameRequest(BaseModel):
    """Mappa SPEAKER_00 -> 'Mario Rossi'."""

    speaker_names: dict[str, str]
    regenerate: bool = Field(
        default=True,
        description="Se true, rigenera riassunto e action point con i nomi reali.",
    )


# ---------------------------------------------------------------------------
# Impostazioni modificabili dall'interfaccia
# ---------------------------------------------------------------------------
class AppSettings(BaseModel):
    """Stato corrente delle impostazioni (il token è sempre mascherato)."""

    hf_token_set: bool = False
    hf_token_preview: str = ""

    whisper_model: str
    whisper_compute_type: str
    whisper_batch_size: int
    whisper_language: str
    diarization_model: str

    # NB: argomento, termini e numero di interlocutori NON stanno qui.
    # Cambiano da una riunione all'altra: vivono sul singolo job.
    beam_size: int = 5
    no_speech_threshold: float = 0.6
    vad_onset: float = 0.5
    vad_offset: float = 0.363
    multilingual: bool = False
    audio_preprocess: bool = False
    auto_context: bool = True
    diarization_fill_nearest: bool = False
    speaker_change_min_words: int = 2
    speaker_change_min_s: float = 0.5
    semantic_search: bool = True
    embed_model: str = "nomic-embed-text"

    ollama_base_url: str
    ollama_model: str
    ollama_num_ctx: int
    ollama_timeout_s: int
    ollama_keep_alive: str = "5m"
    chunk_chars: int

    max_upload_mb: int
    delete_audio_after: bool

    # Campi sovrascritti da interfaccia (gli altri vengono da .env)
    overridden: list[str] = Field(default_factory=list)


class SettingsPatch(BaseModel):
    """Aggiornamento parziale: solo i campi effettivamente inviati vengono applicati.

    Usa `model_dump(exclude_unset=True)` per distinguere "non inviato" da
    "inviato a null" (che significa: azzera il valore).
    """

    model_config = {"extra": "forbid"}

    hf_token: str | None = None

    whisper_model: str | None = None
    whisper_compute_type: str | None = None
    whisper_batch_size: int | None = Field(default=None, ge=1, le=64)
    whisper_language: str | None = None
    diarization_model: str | None = None
    beam_size: int | None = Field(default=None, ge=1, le=10)
    no_speech_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    vad_onset: float | None = Field(default=None, ge=0.05, le=0.95)
    vad_offset: float | None = Field(default=None, ge=0.05, le=0.95)
    multilingual: bool | None = None
    audio_preprocess: bool | None = None
    auto_context: bool | None = None
    diarization_fill_nearest: bool | None = None
    speaker_change_min_words: int | None = Field(default=None, ge=1, le=12)
    speaker_change_min_s: float | None = Field(default=None, ge=0.0, le=5.0)
    semantic_search: bool | None = None
    embed_model: str | None = Field(default=None, max_length=120)

    ollama_base_url: str | None = None
    ollama_model: str | None = None
    ollama_num_ctx: int | None = Field(default=None, ge=2048, le=262144)
    ollama_timeout_s: int | None = Field(default=None, ge=30, le=7200)
    ollama_keep_alive: str | None = Field(default=None, max_length=16)
    chunk_chars: int | None = Field(default=None, ge=2000, le=400000)

    max_upload_mb: int | None = Field(default=None, ge=1, le=20480)
    delete_audio_after: bool | None = None


class SettingsOptions(BaseModel):
    """Valori proposti dalla UI nelle tendine."""

    whisper_models: list[dict]
    compute_types: list[dict]
    languages: list[dict]
    diarization_models: list[str]
    suggested_llms: list[dict]
    installed_llms: list[str]


class PullRequest(BaseModel):
    model: str = Field(min_length=1)
