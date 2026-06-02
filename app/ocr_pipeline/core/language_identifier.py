"""Multi-layer language identification with optional fastText and CLD3."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from app.config import LanguageIdentifierConfig
from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)

_RTL_LANGS = frozenset({"ar", "ara", "fa", "fas", "he", "heb", "ur", "urd", "yi", "yid"})


@dataclass(slots=True)
class LanguageIdentificationResult:
    primary_language: str
    direction: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    method: str = "rule_based"


def _direction(lang: str) -> str:
    return "rtl" if lang.lower() in _RTL_LANGS else "ltr"


class LanguageIdentifier:
    """Three-layer language identification with automatic fallback."""

    def __init__(self, config: LanguageIdentifierConfig | None = None) -> None:
        self.config = config or LanguageIdentifierConfig()
        self._ft_model = None
        self._ft_attempted = False
        self._cld3_available: Optional[bool] = None

    def identify(self, text: str) -> LanguageIdentificationResult:
        if not text or not text.strip():
            return LanguageIdentificationResult("unknown", "ltr", 0.0)

        result = self._fasttext(text) if self.config.use_fasttext else None
        if result is not None:
            return result

        result = self._cld3(text) if self.config.use_cld3 else None
        if result is not None:
            return result

        return self._rule_based(text)

    def _fasttext(self, text: str) -> Optional[LanguageIdentificationResult]:
        model = self._load_fasttext()
        if model is None:
            return None
        try:
            clean = text.replace("\n", " ")[:1000]
            labels, scores = model.predict(clean, k=3)
            lang = labels[0].replace("__label__", "")
            probs = {label.replace("__label__", ""): float(score) for label, score in zip(labels, scores)}
            return LanguageIdentificationResult(
                primary_language=lang,
                direction=_direction(lang),
                confidence=round(float(scores[0]), 4),
                probabilities=probs,
                method="fasttext",
            )
        except Exception as exc:
            LOGGER.warning("fastText identify failed: %s", exc)
            return None

    def _load_fasttext(self):
        if self._ft_attempted:
            return self._ft_model
        self._ft_attempted = True
        try:
            import fasttext  # type: ignore

            search = [
                Path(self.config.fasttext_model),
                Path("models/lid.176.bin"),
                Path.home() / ".cache" / "fasttext" / "lid.176.ftz",
            ]
            for path in search:
                if path.exists():
                    self._ft_model = fasttext.load_model(str(path))
                    LOGGER.info("fastText model loaded: %s", path)
                    return self._ft_model
        except ImportError:
            return None
        except Exception as exc:
            LOGGER.warning("fastText load failed: %s", exc)
        return None

    def _cld3(self, text: str) -> Optional[LanguageIdentificationResult]:
        if self._cld3_available is False:
            return None
        try:
            import gcld3  # type: ignore

            self._cld3_available = True
            detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=512)
            result = detector.FindLanguage(text=text[:512])
            if not result.is_reliable:
                return None
            return LanguageIdentificationResult(
                primary_language=result.language,
                direction=_direction(result.language),
                confidence=round(float(result.probability), 4),
                probabilities={result.language: float(result.probability)},
                method="cld3",
            )
        except ImportError:
            self._cld3_available = False
            return None
        except Exception as exc:
            LOGGER.warning("CLD3 identify failed: %s", exc)
            return None

    def _rule_based(self, text: str) -> LanguageIdentificationResult:
        from app.ocr_pipeline.language.detector import LanguageDetector

        profile = LanguageDetector().detect(text)
        return LanguageIdentificationResult(
            primary_language=profile.language,
            direction=profile.direction,
            confidence=float(profile.probabilities.get(profile.language, 0.5)),
            probabilities=profile.probabilities,
            method="rule_based",
        )

