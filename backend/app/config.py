"""Configurazione centralizzata, letta da variabili d'ambiente / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _blank_means_unset(cls, data: Any) -> Any:
        """`MIN_SPEAKERS=` nel .env significa "non impostato", non "stringa vuota".

        Docker Compose passa le variabili dichiarate ma vuote come `''`, che
        Pydantic non sa convertire in intero. Per i campi non testuali scartiamo
        il valore e lasciamo agire il default; per quelli testuali lo teniamo,
        perché lì la stringa vuota è significativa (es. lingua = automatica).
        """
        if not isinstance(data, dict):
            return data

        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str) and not value.strip():
                field = cls.model_fields.get(key) or cls.model_fields.get(str(key).lower())
                if field is not None and field.annotation is not str:
                    continue
            cleaned[key] = value
        return cleaned

    # --- Percorsi ---
    data_dir: Path = Path("/data")

    # --- Hugging Face ---
    hf_token: str = ""

    # --- WhisperX ---
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "float16"
    whisper_batch_size: int = 16
    whisper_language: str = "it"
    diarization_model: str = "pyannote/speaker-diarization-community-1"

    # --- Qualità della trascrizione ---
    # NB: argomento, termini e numero di interlocutori NON stanno qui.
    # Sono proprietà della singola riunione e si impostano al caricamento
    # del file: vedi `models.JobOptions`.
    beam_size: int = 5
    #: Sotto questa probabilità un chunk è considerato silenzio.
    no_speech_threshold: float = 0.6
    #: Sensibilità del rilevamento del parlato: più basso = più permissivo.
    vad_onset: float = 0.500
    vad_offset: float = 0.363
    #: Rilevamento della lingua per segmento: utile su audio che mescola lingue.
    multilingual: bool = False

    # --- Attribuzione delle voci ---
    #: Attribuisce alla voce temporalmente più vicina anche le parole per cui
    #: la diarizzazione non ha alcuna evidenza (pause, respiri, parole a
    #: cavallo fra due turni). Copre tutto, ma inventa: è la causa principale
    #: dei falsi cambi di parlante. Attivalo solo se restano segmenti anonimi.
    diarization_fill_nearest: bool = False
    #: Un cambio di voce che dura meno di così è rumore di confine, non un
    #: turno: viene riassorbito da chi stava parlando. Basta superare UNA
    #: delle due soglie perché il cambio sia accettato.
    speaker_change_min_words: int = 2
    speaker_change_min_s: float = 0.5
    #: Filtro passa-alto + riduzione rumore + normalizzazione prima dell'ASR.
    audio_preprocess: bool = False
    #: Ricognizione automatica: un primo passaggio veloce su alcuni estratti
    #: deduce argomento e vocabolario, che vengono usati per la trascrizione vera.
    auto_context: bool = True

    # --- Separazione delle voci sovrapposte (servizio opzionale) ---
    separator_url: str = "http://separator:8000"
    separator_timeout_s: int = 3600
    #: Sotto questa percentuale di parlato sovrapposto la separazione non vale
    #: il tempo che costa: il risultato sarebbe quasi identico.
    separation_min_overlap_pct: float = 1.0

    # --- Ollama ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:8b"
    ollama_num_ctx: int = 32768
    ollama_timeout_s: int = 900
    #: Quanto il modello resta in VRAM dopo la risposta. Viaggia nella singola
    #: richiesta, non come variabile d'ambiente del container: così è
    #: modificabile a caldo senza riavviare Ollama.
    #: "0" = scarica subito · "5m" = cinque minuti · "-1" = sempre in memoria.
    ollama_keep_alive: str = "5m"
    chunk_chars: int = 12000

    # --- Ricerca ---
    #: Ricerca per significato oltre a quella per parole esatte. Richiede un
    #: modello di embedding su Ollama; se manca, la ricerca continua a
    #: funzionare sulle sole parole esatte.
    semantic_search: bool = True
    embed_model: str = "nomic-embed-text"

    # --- App ---
    max_upload_mb: int = 1024
    delete_audio_after: bool = False

    # ------------------------------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "timbro.sqlite3"

    @property
    def language(self) -> str | None:
        """`None` = rilevamento automatico della lingua."""
        lang = (self.whisper_language or "").strip().lower()
        return None if lang in ("", "auto") else lang

    def asr_options(self, prompt: str = "", hotwords: str = "") -> dict:
        """Opzioni passate a `whisperx.load_model(asr_options=...)`.

        Includiamo solo le chiavi che vogliamo davvero sovrascrivere: WhisperX
        fa il merge con i propri default, quindi il resto resta invariato.
        Prompt e hotwords arrivano dal job, non dalla configurazione globale.
        """
        options: dict = {
            "beam_size": int(self.beam_size),
            "best_of": int(self.beam_size),
            "no_speech_threshold": float(self.no_speech_threshold),
            "multilingual": bool(self.multilingual),
        }
        if prompt := (prompt or "").strip():
            options["initial_prompt"] = prompt
        if hot := (hotwords or "").strip():
            options["hotwords"] = hot
        return options

    @property
    def vad_options(self) -> dict:
        return {"vad_onset": float(self.vad_onset), "vad_offset": float(self.vad_offset)}

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
