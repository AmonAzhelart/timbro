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


class Segment(BaseModel):
    start: float
    end: float
    speaker: str = "SPEAKER_?"
    text: str
    #: Voci attive contemporaneamente in questo tratto. Se valorizzato, il
    #: testo è meno affidabile: Whisper trascrive un solo flusso.
    overlap: list[str] = Field(default_factory=list)
    #: True se questo testo proviene da una traccia audio separata, quindi
    #: appartiene davvero a questa voce e non al parlante dominante.
    separated: bool = False


class OverlapStats(BaseModel):
    count: int = 0
    total_s: float = 0.0
    percent: float = 0.0
    max_speakers: int = 0
    separated: bool = False


class ActionPoint(BaseModel):
    task: str = Field(description="Azione concreta da svolgere")
    owner: str = Field(default="Non assegnato", description="Chi se ne occupa")
    due: str = Field(default="Non specificata", description="Scadenza indicata")
    priority: str = Field(default="media", description="alta | media | bassa")


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
    speakers: list[str] = Field(default_factory=list)
    title: str | None = None
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
