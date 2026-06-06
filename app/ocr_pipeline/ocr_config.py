"""
DEPRECATED: Central configuration for the local document intelligence pipeline.
Use app.config instead.
"""

from __future__ import annotations
from app.config import get_settings

# Bridge to new config system
_settings = get_settings()
APP_CONFIG = _settings.OCR_PIPELINE

# Legacy constants mapping for backward compatibility
# These should be removed eventually as files are updated to use settings directly.

OCR_USE_GPU = APP_CONFIG.ocr.use_gpu
OCR_LANG_EN = "en"
OCR_LANG_FR = "fr"
OCR_LANG_AR = "ar"
OCR_USE_ANGLE_CLS = APP_CONFIG.ocr.use_angle_cls

LAYOUT_ENABLED = APP_CONFIG.layout.enabled
LAYOUT_USE_GPU = APP_CONFIG.layout.use_gpu

ALLOWED_INPUT_EXTENSIONS = APP_CONFIG.input.allowed_extensions

QWEN_ENABLED = APP_CONFIG.qwen.enabled
QWEN_MODEL = APP_CONFIG.qwen.model
OLLAMA_BASE_URL = APP_CONFIG.qwen.base_url
QWEN_TEMPERATURE = APP_CONFIG.qwen.temperature
QWEN_TIMEOUT = APP_CONFIG.qwen.timeout_sec

OUTPUT_DIR = APP_CONFIG.output_dir
API_ALLOWED_ORIGINS = APP_CONFIG.api_allowed_origins
API_CLEANUP_UPLOADS = APP_CONFIG.cleanup_uploads
CORE_MODE = APP_CONFIG.core_mode

SUPPORTED_LANGUAGES = APP_CONFIG.language.supported_languages

OCR_SCORE_CONFIDENCE_WEIGHT = APP_CONFIG.ocr.confidence_weight
OCR_SCORE_DENSITY_WEIGHT = APP_CONFIG.ocr.density_weight
OCR_SCORE_DENSITY_LOG_BASE = APP_CONFIG.ocr.density_log_base

OCR_IOU_DUPLICATE_THRESHOLD = APP_CONFIG.ocr.iou_duplicate_threshold
OCR_CLOSE_Y_RATIO = APP_CONFIG.ocr.close_y_ratio
OCR_CLOSE_X_RATIO = APP_CONFIG.ocr.close_x_ratio

COL_SEP_GAP_WEIGHT = APP_CONFIG.layout.col_sep_gap_weight
COL_SEP_BALANCE_WEIGHT = APP_CONFIG.layout.col_sep_balance_weight
COL_SEP_CROSSING_PENALTY = APP_CONFIG.layout.col_sep_crossing_penalty
COL_SEP_SEARCH_START_RATIO = APP_CONFIG.layout.col_sep_search_start_ratio
COL_SEP_SEARCH_STOP_RATIO = APP_CONFIG.layout.col_sep_search_stop_ratio
COL_SEP_MIN_GAP_ABS = APP_CONFIG.layout.col_sep_min_gap_abs
COL_SEP_MIN_GAP_RATIO = APP_CONFIG.layout.col_sep_min_gap_ratio
COL_SEP_SIDEBAR_SPAN_RATIO = APP_CONFIG.layout.col_sep_sidebar_span_ratio
COL_SEP_SIDEBAR_GROWTH_RATIO = APP_CONFIG.layout.col_sep_sidebar_growth_ratio
COL_SEP_SIDEBAR_X_RATIO = APP_CONFIG.layout.col_sep_sidebar_x_ratio

# Type aliases for backward compatibility
AppConfig = type(APP_CONFIG)
