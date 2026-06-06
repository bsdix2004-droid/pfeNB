"""
app/config.py

Single configuration system for Scanalyze.

Load order (highest precedence first):
  1) Environment variables
  2) .env file (dotenv)
  3) app/ocr_pipeline/configs/default.json (fallback)

Includes a CLI helper:
  python -m app.config show
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Set Paddle/GLog environment variables immediately on import
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_logtostderr", "0")
os.environ.setdefault("FLAGS_minloglevel", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("PADDLE_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("KMP_WARNINGS", "0")


DEFAULT_OCR_CONFIG_PATH = Path(__file__).resolve().parent / "ocr_pipeline" / "configs" / "default.json"


class OCRInputConfig(BaseModel):
    allowed_extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".svg", ".pdf"])
    max_file_size_mb: int = 25
    min_width: int = 80
    min_height: int = 80
    max_width: int = 6000
    max_height: int = 6000
    max_pixels: int = 25_000_000
    normalize_max_long_edge: int = 3200
    svg_render_width: int = 1800


class OCRPreprocessingConfig(BaseModel):
    enabled: bool = True
    min_width: int = 1200
    max_width: int = 2200
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    adaptive_block_size: int = 31
    adaptive_c: int = 11
    denoise_blur_threshold: float = 80.0
    noise_threshold: float = 9.0
    contrast_threshold: float = 45.0
    dark_threshold: float = 80.0
    bright_threshold: float = 210.0
    deskew_enabled: bool = True
    perspective_correction_enabled: bool = True
    shadow_correction_enabled: bool = True
    orientation_candidates_enabled: bool = True
    multi_pass_enabled: bool = True
    border_cleanup_margin: int = 6
    max_candidates: int = 6


class ScriptDetectorConfig(BaseModel):
    visual_weight: float = 0.30
    unicode_weight: float = 0.70
    min_script_ratio: float = 0.15


class OCREngineConfig(BaseModel):
    use_angle_cls: bool = True
    use_gpu: bool = False
    ocr_version: str = "PP-OCRv5"
    min_confidence: float = 0.70
    languages: list[str] = Field(default_factory=lambda: ["en", "fr", "ar"])
    low_confidence_threshold: float = 0.70
    candidate_selection: bool = True
    max_ocr_candidates: int = 4
    worker_timeout_sec: int = 240
    multilingual_fallback_enabled: bool = True
    multilingual_fallback_min_lines: int = 4
    multilingual_fallback_min_chars: int = 40
    multilingual_fallback_min_confidence: float = 0.60
    
    # Scoring weights from ocr_config.py
    confidence_weight: float = 0.72
    density_weight: float = 0.28
    density_log_base: int = 1200
    
    # Duplicate detection
    iou_duplicate_threshold: float = 0.65
    close_y_ratio: float = 0.55
    close_x_ratio: float = 0.35


class LayoutConfig(BaseModel):
    enabled: bool = True
    use_gpu: bool = False
    timeout_sec: int = 300
    
    # Column separator scoring
    col_sep_gap_weight: float = 2.4
    col_sep_balance_weight: float = 0.25
    col_sep_crossing_penalty: float = 1.8
    col_sep_search_start_ratio: float = 0.18
    col_sep_search_stop_ratio: float = 0.58
    col_sep_min_gap_abs: float = 45.0
    col_sep_min_gap_ratio: float = 0.035
    col_sep_sidebar_span_ratio: float = 0.38
    col_sep_sidebar_growth_ratio: float = 1.25
    col_sep_sidebar_x_ratio: float = 0.30


class LanguageIdentifierConfig(BaseModel):
    use_fasttext: bool = True
    use_cld3: bool = True
    fasttext_model: str = "models/lid.176.ftz"
    fasttext_model_url: str = (
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    )
    fallback_to_rules: bool = True
    supported_languages: list[str] = Field(default_factory=lambda: ["ar", "en", "fr", "zh"])


class EmbeddingConfig(BaseModel):
    enabled: bool = True
    model_name: str = "all-MiniLM-L6-v2"
    use_cache: bool = True
    faiss_threshold: int = 200


class QwenConfig(BaseModel):
    enabled: bool = True
    model: str = "qwen2.5:0.5b"
    base_url: str = "http://ollama:11434"
    temperature: float = 0.0
    timeout_sec: int = 60
    top_p: float = 0.1
    min_ocr_confidence: float = 0.85
    context_length: int = 1024
    max_output_tokens: int = 512
    num_gpu: int = 0


class OCRDetectorConfig(BaseModel):
    # If the classifier is unsure, fall back to "unknown" rather than guessing.
    min_confidence: float = 0.20


class OCRPipelineConfig(BaseModel):
    input: OCRInputConfig = Field(default_factory=OCRInputConfig)
    preprocessing: OCRPreprocessingConfig = Field(default_factory=OCRPreprocessingConfig)
    script: ScriptDetectorConfig = Field(default_factory=ScriptDetectorConfig)
    ocr: OCREngineConfig = Field(default_factory=OCREngineConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    language: LanguageIdentifierConfig = Field(default_factory=LanguageIdentifierConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qwen: QwenConfig = Field(default_factory=QwenConfig)
    detector: OCRDetectorConfig = Field(default_factory=OCRDetectorConfig)

    output_dir: str = "output"
    api_allowed_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:3000", "http://localhost:3000", "http://127.0.0.1:5173", "http://localhost:5173"])
    cleanup_uploads: bool = True
    core_mode: bool = False


def _load_default_ocr_pipeline_config() -> dict[str, Any]:
    """
    Read app/ocr_pipeline/configs/default.json and map it into OCRPipelineConfig fields.
    Missing keys simply use OCRPipelineConfig defaults.
    """

    if not DEFAULT_OCR_CONFIG_PATH.exists():
        return {}
    try:
        raw = json.loads(DEFAULT_OCR_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
        
    mapped: dict[str, Any] = {}

    preprocessing = raw.get("preprocessing") if isinstance(raw, dict) else {}
    if isinstance(preprocessing, dict):
        mapped.setdefault("preprocessing", {})
        mapped["preprocessing"]["enabled"] = bool(preprocessing.get("adaptive", True))
        mapped["preprocessing"]["deskew_enabled"] = bool(preprocessing.get("deskew", True))
        mapped["preprocessing"]["orientation_candidates_enabled"] = bool(preprocessing.get("orientation_candidates", True))
        mapped["preprocessing"]["max_candidates"] = int(preprocessing.get("max_candidates", 6))

    limits = raw.get("input_limits") if isinstance(raw, dict) else {}
    if isinstance(limits, dict):
        mapped.setdefault("input", {})
        mapped["input"]["allowed_extensions"] = list(limits.get("allowed_extensions", OCRInputConfig().allowed_extensions))
        mapped["input"]["max_file_size_mb"] = int(limits.get("max_file_size_mb", 25))
        min_res = limits.get("min_resolution", [80, 80])
        max_res = limits.get("max_resolution", [6000, 6000])
        if isinstance(min_res, list) and len(min_res) >= 2:
            mapped["input"]["min_width"] = int(min_res[0])
            mapped["input"]["min_height"] = int(min_res[1])
        if isinstance(max_res, list) and len(max_res) >= 2:
            mapped["input"]["max_width"] = int(max_res[0])
            mapped["input"]["max_height"] = int(max_res[1])
        mapped["input"]["max_pixels"] = int(limits.get("max_pixels", 25_000_000))
        mapped["input"]["normalize_max_long_edge"] = int(limits.get("normalize_max_long_edge", 3200))
        mapped["input"]["svg_render_width"] = int(limits.get("svg_render_width", 1800))

    ocr = raw.get("ocr") if isinstance(raw, dict) else {}
    if isinstance(ocr, dict):
        mapped.setdefault("ocr", {})
        mapped["ocr"]["ocr_version"] = str(ocr.get("primary", "PP-OCRv5"))
        mapped["ocr"]["languages"] = list(ocr.get("languages", ["en", "fr", "ar"]))
        mapped["ocr"]["use_gpu"] = not bool(ocr.get("cpu_only", True))
        mapped["ocr"]["min_confidence"] = float(ocr.get("min_confidence", 0.70))
        mapped["ocr"]["low_confidence_threshold"] = float(ocr.get("low_confidence_threshold", 0.70))
        mapped["ocr"]["max_ocr_candidates"] = int(ocr.get("max_ocr_candidates", 4))
        mapped["ocr"]["worker_timeout_sec"] = int(ocr.get("worker_timeout_sec", 240))

    layout = raw.get("layout") if isinstance(raw, dict) else {}
    if isinstance(layout, dict):
        mapped.setdefault("layout", {})
        mapped["layout"]["use_gpu"] = not bool(layout.get("cpu_only", True))
        mapped["layout"]["timeout_sec"] = int(layout.get("timeout_sec", 45))

    ai = raw.get("ai") if isinstance(raw, dict) else {}
    if isinstance(ai, dict):
        mapped.setdefault("qwen", {})
        mapped["qwen"]["temperature"] = float(ai.get("temperature", 0.0))
        mapped["qwen"]["top_p"] = float(ai.get("top_p", 0.1))
        mapped["qwen"]["min_ocr_confidence"] = float(ai.get("min_ocr_confidence", 0.85))
        mapped["qwen"]["context_length"] = int(ai.get("context_length", 1024))
        mapped["qwen"]["max_output_tokens"] = int(ai.get("max_output_tokens", 512))
        mapped["qwen"]["timeout_sec"] = int(ai.get("timeout_sec", 60))
        mapped["qwen"]["num_gpu"] = int(ai.get("num_gpu", 0))

    return mapped


def _default_json_settings_source(**kwargs: Any) -> dict[str, Any]:
    return {"OCR_PIPELINE": _load_default_ocr_pipeline_config()}


class Settings(BaseSettings):
    # ── Core ──────────────────────────────────
    APP_ENV: Literal["dev", "staging", "prod", "test"] = "dev"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "dev-secret-key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALLOWED_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # ── Database ──────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "postgres"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # ── Redis / Celery ────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── Ollama (legacy app-level defaults) ────
    OLLAMA_HOST: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:0.5b"
    OLLAMA_TIMEOUT: int = 120

    # ── OCR Engine (legacy orchestrator) ──────
    OCR_ENGINE_URL: str = "http://localhost:8001"

    # ── Object Storage ────────────────────────
    S3_ENDPOINT_URL: str | None = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_UPLOADS: str = "vision-uploads"
    S3_BUCKET_RESULTS: str = "vision-results"
    S3_REGION: str = "us-east-1"

    # ── Email ─────────────────────────────────
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = "noreply@example.com"
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = 24
    FRONTEND_URL: str = "http://localhost:3000"

    # ── Monitoring ────────────────────────────
    SENTRY_DSN: str = ""
    SENTRY_TRACES_SAMPLE_RATE: float = 0.0

    # ── Rate limiting ─────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60

    # ── OCR Pipeline (intelligent document analyzer) ───────
    OCR_PIPELINE: OCRPipelineConfig = Field(default_factory=OCRPipelineConfig)

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_dev(self) -> bool:
        return self.APP_ENV == "dev"

    @property
    def is_prod(self) -> bool:
        return self.APP_ENV == "prod"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Order matters: env vars > .env > default.json > init kwargs
        return (env_settings, dotenv_settings, _default_json_settings_source, init_settings, file_secret_settings)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — called via FastAPI Depends and workers."""
    return Settings()


def _show() -> None:
    settings = get_settings()
    print(json.dumps(settings.model_dump(mode="json"), indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    import sys

    cmd = (sys.argv[1:] or ["show"])[0]
    if cmd != "show":
        raise SystemExit("Usage: python -m app.config show")
    _show()


if __name__ == "__main__":
    main()
