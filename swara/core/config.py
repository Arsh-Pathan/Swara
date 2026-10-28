"""Centralized configuration for Swara."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for Swara."""

    model_config = SettingsConfigDict(
        env_prefix="SWARA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Paths
    base_dir: Path = Field(default_factory=lambda: Path.cwd())
    models_dir: Path = Field(default=Path("models/cache"))
    speaker_store_path: Path = Field(default=Path("data/speakers"))

    # Audio Pipeline
    target_sample_rate: int = 16000
    max_audio_duration_seconds: float = 1800.0  # 30 minutes
    max_audio_upload_size_mb: float = 50.0

    # VAD Configuration
    vad_onset_threshold: float = 0.50
    vad_offset_threshold: float = 0.35
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 200
    vad_speech_pad_ms: int = 30

    # Speaker Embedding & Model Backend
    use_int8_model: bool = False
    execution_provider: str = "CPUExecutionProvider"  # or 'CUDAExecutionProvider'
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 2

    # Speaker Identification Thresholds
    # In VoxCeleb evaluation, cosine similarity > 0.68 is optimal balance between FAR and FRR
    identification_threshold: float = 0.68
    unknown_threshold: float = 0.60
    min_enrollment_duration_s: float = 0.8
    min_identification_duration_s: float = 0.5

    # Speaker Diarization
    diarization_clustering_threshold: float = 0.65
    diarization_window_size_s: float = 1.5
    diarization_step_size_s: float = 0.75
    diarization_collar_s: float = 0.35
    diarization_min_speakers: Optional[int] = None
    diarization_max_speakers: Optional[int] = 10
    diarization_identity_match_threshold: float = 0.70

    # API & Security
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: Optional[str] = None
    cors_origins: list[str] = ["*"]
    rate_limit_per_minute: int = 120
    log_level: str = "INFO"

    def resolve_paths(self) -> None:
        """Resolve all paths relative to base_dir if not absolute."""
        if not self.models_dir.is_absolute():
            self.models_dir = (self.base_dir / self.models_dir).resolve()
        if not self.speaker_store_path.is_absolute():
            self.speaker_store_path = (self.base_dir / self.speaker_store_path).resolve()


# Singleton instance
settings = Settings()
settings.resolve_paths()
