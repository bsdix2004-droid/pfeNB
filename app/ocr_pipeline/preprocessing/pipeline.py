"""Adaptive OpenCV preprocessing for document OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.config import OCRPreprocessingConfig as PreprocessingConfig
from app.ocr_pipeline.preprocessing.quality import ImageQualityAnalyzer, ImageQualityProfile
from app.ocr_pipeline.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class PreprocessedCandidate:
    name: str
    image: np.ndarray
    score: float
    steps: List[str] = field(default_factory=list)
    quality: Dict[str, object] = field(default_factory=dict)

    def to_dict(self, include_quality: bool = True) -> Dict[str, object]:
        data: Dict[str, object] = {
            "name": self.name,
            "score": round(float(self.score), 4),
            "steps": list(self.steps),
            "shape": list(self.image.shape),
        }
        if include_quality:
            data["quality"] = self.quality
        return data


@dataclass(slots=True)
class PreprocessingResult:
    selected_index: int
    candidates: List[PreprocessedCandidate]
    original_quality: Dict[str, object]
    operations: List[str] = field(default_factory=list)

    @property
    def selected_candidate(self) -> PreprocessedCandidate:
        return self.candidates[self.selected_index]

    @property
    def image(self) -> np.ndarray:
        return self.selected_candidate.image

    def select_candidate(self, candidate_index: int) -> None:
        if 0 <= candidate_index < len(self.candidates):
            self.selected_index = candidate_index

    def to_dict(self) -> Dict[str, object]:
        return {
            "selected_candidate": self.selected_candidate.name,
            "selected_index": self.selected_index,
            "candidate_count": len(self.candidates),
            "operations": list(self.operations),
            "original_quality": self.original_quality,
            "candidates": [candidate.to_dict(include_quality=True) for candidate in self.candidates],
        }


def analyze_image(image: np.ndarray) -> Dict[str, object]:
    analyzer = ImageQualityAnalyzer()
    profile = analyzer.analyze(image)
    data = profile.to_dict()
    data.update(
        {
            "needs_upscale": profile.width < 1200,
            "needs_downscale": profile.width > 2200,
            "needs_denoise": profile.noise_score > 9.0 and profile.blur_score > 65.0,
            "needs_contrast_boost": profile.contrast < 45.0 or profile.brightness < 80.0 or profile.brightness > 210.0,
            "needs_deskew": abs(profile.skew_angle) > 0.35,
            "needs_shadow_correction": profile.shadow_score > 0.08,
            "needs_border_cleanup": profile.border_score > 0.08,
            "needs_perspective_correction": profile.perspective_confidence > 0.55,
        }
    )
    return data


def deskew(gray: np.ndarray) -> np.ndarray:
    corrected, _angle = deskew_with_angle(gray)
    return corrected


def deskew_with_angle(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    profile_angle = ImageQualityAnalyzer().estimate_skew_angle(gray)
    if abs(profile_angle) < 0.35:
        return gray, 0.0
    return _rotate_image(gray, profile_angle), float(profile_angle)


def clean_borders(image: np.ndarray, margin: int = 6) -> np.ndarray:
    cleaned = image.copy()
    if margin <= 0:
        return cleaned
    cleaned[:margin, :] = 255
    cleaned[-margin:, :] = 255
    cleaned[:, :margin] = 255
    cleaned[:, -margin:] = 255
    return cleaned


def normalize_for_ocr(image: np.ndarray, config: PreprocessingConfig | None = None) -> Tuple[np.ndarray, Dict[str, object], List[str]]:
    """Backward-compatible normalized grayscale image."""
    result = DocumentPreprocessor(config or PreprocessingConfig()).process_with_metadata(image)
    for candidate in result.candidates:
        if candidate.name == "enhanced_gray":
            return candidate.image, result.original_quality, candidate.steps
    return result.image, result.original_quality, result.selected_candidate.steps


def adaptive_preprocess(image: np.ndarray, config: PreprocessingConfig | None = None) -> Tuple[np.ndarray, Dict[str, object], List[str]]:
    """Backward-compatible adaptive binary image."""
    result = DocumentPreprocessor(config or PreprocessingConfig()).process_with_metadata(image)
    for candidate in result.candidates:
        if candidate.name in {"sauvola_binary", "adaptive_binary"}:
            return candidate.image, result.original_quality, candidate.steps
    return result.image, result.original_quality, result.selected_candidate.steps


class DocumentPreprocessor:
    """Analyze image quality, then create OCR candidates that preserve layout."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.quality_analyzer = ImageQualityAnalyzer()

    def process(self, image: np.ndarray) -> np.ndarray:
        return self.process_with_metadata(image).image

    def process_with_metadata(self, image: np.ndarray) -> PreprocessingResult:
        if image is None or image.size == 0:
            raise ValueError("Cannot preprocess an empty image")

        if not self.config.enabled:
            quality = self._quality_metadata(image)
            candidate = PreprocessedCandidate("original", image.copy(), self._score_image(image), ["preprocessing_disabled"], quality)
            return PreprocessingResult(0, [candidate], quality, ["preprocessing_disabled"])

        original_quality = self._quality_profile(image)
        normalized_color, base_steps = self._prepare_color_image(image, original_quality)
        gray, gray_steps = self._prepare_gray_image(normalized_color, original_quality)

        candidates = [
            self._candidate("color_preserved", normalized_color, base_steps),
            self._candidate("enhanced_gray", gray, base_steps + gray_steps),
        ]

        if self.config.orientation_candidates_enabled:
            candidates.extend(self._orientation_candidates(normalized_color, gray, base_steps + gray_steps))

        if self.config.multi_pass_enabled:
            candidates.extend(self._binary_candidates(gray, original_quality, base_steps + gray_steps))

        unique: Dict[str, PreprocessedCandidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.name, candidate)
        ordered = sorted(unique.values(), key=lambda item: item.score, reverse=True)
        limited = ordered[: max(1, self.config.max_candidates)]
        operations = sorted({step for candidate in limited for step in candidate.steps})
        return PreprocessingResult(0, limited, original_quality.to_dict(), operations)

    def _prepare_color_image(self, image: np.ndarray, profile: ImageQualityProfile) -> Tuple[np.ndarray, List[str]]:
        steps: List[str] = []
        color = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if self.config.perspective_correction_enabled and profile.perspective_confidence > 0.55:
            corrected, changed = self._perspective_correct(color)
            if changed:
                color = corrected
                steps.append("perspective_correct")

        color, scale_step = self._normalize_resolution(color)
        if scale_step:
            steps.append(scale_step)

        if profile.border_score > 0.08:
            color = self._trim_dark_borders(color)
            steps.append("border_trim")

        return color, steps

    def _prepare_gray_image(self, color: np.ndarray, profile: ImageQualityProfile) -> Tuple[np.ndarray, List[str]]:
        steps: List[str] = []
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color.copy()

        if self.config.shadow_correction_enabled and profile.shadow_score > 0.08:
            gray = self._correct_shadows(gray)
            steps.append("shadow_correct")

        if profile.contrast < self.config.contrast_threshold or profile.brightness < self.config.dark_threshold or profile.brightness > self.config.bright_threshold:
            clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip_limit, tileGridSize=self.config.clahe_tile_grid_size)
            gray = clahe.apply(gray)
            steps.append("clahe")
        else:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            steps.append("histogram_normalize")

        if profile.noise_score > self.config.noise_threshold and profile.blur_score > self.config.denoise_blur_threshold:
            gray = cv2.fastNlMeansDenoising(gray, h=9, templateWindowSize=7, searchWindowSize=21)
            steps.append("nl_means_denoise")
        elif profile.noise_score > self.config.noise_threshold * 0.75:
            gray = cv2.bilateralFilter(gray, 7, 40, 40)
            steps.append("bilateral_denoise")

        if self.config.deskew_enabled:
            gray, angle = deskew_with_angle(gray)
            steps.append(f"deskew:{angle:.2f}" if abs(angle) >= 0.35 else "deskew_checked")

        if profile.border_score > 0.08:
            gray = clean_borders(gray, self.config.border_cleanup_margin)
            steps.append("border_cleanup")

        return gray, steps

    def _binary_candidates(
        self,
        gray: np.ndarray,
        profile: ImageQualityProfile,
        parent_steps: List[str],
    ) -> List[PreprocessedCandidate]:
        candidates: List[PreprocessedCandidate] = []
        block_size = max(15, int(self.config.adaptive_block_size))
        if block_size % 2 == 0:
            block_size += 1

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.config.adaptive_c,
        )
        adaptive = self._light_morphology(adaptive)
        candidates.append(self._candidate("adaptive_binary", adaptive, parent_steps + ["adaptive_threshold", "light_morphology"]))

        sauvola = self._sauvola(gray)
        if sauvola is not None:
            sauvola = self._light_morphology(sauvola)
            candidates.append(self._candidate("sauvola_binary", sauvola, parent_steps + ["sauvola_threshold", "light_morphology"]))

        if profile.shadow_score < 0.08 and profile.contrast >= self.config.contrast_threshold:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidates.append(self._candidate("otsu_binary", otsu, parent_steps + ["otsu_threshold"]))

        return candidates

    def _orientation_candidates(self, color: np.ndarray, gray: np.ndarray, parent_steps: List[str]) -> List[PreprocessedCandidate]:
        """Add OCR-selectable portrait candidates for sideways phone captures."""
        height, width = gray.shape[:2]
        if width <= height * 1.12:
            return []

        candidates: List[PreprocessedCandidate] = []
        rotations = [
            ("rotate_90_clockwise", cv2.ROTATE_90_CLOCKWISE),
            ("rotate_90_counterclockwise", cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        for name, rotation in rotations:
            rotated = cv2.rotate(color, rotation)
            rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if rotated.ndim == 3 else rotated
            candidates.append(self._candidate(f"orientation_{name}", rotated_gray, parent_steps + [name]))
        return candidates

    def _candidate(self, name: str, image: np.ndarray, steps: List[str]) -> PreprocessedCandidate:
        quality = self._quality_metadata(image)
        return PreprocessedCandidate(name=name, image=image, score=self._score_image(image), steps=list(dict.fromkeys(steps)), quality=quality)

    def _quality_profile(self, image: np.ndarray) -> ImageQualityProfile:
        return self.quality_analyzer.analyze(image)

    def _quality_metadata(self, image: np.ndarray) -> Dict[str, object]:
        try:
            detailed = self.quality_analyzer.analyze(image).to_dict()
        except Exception as exc:
            LOGGER.warning("quality metadata analysis failed: %s", exc)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            detailed = {
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                "brightness": float(np.mean(gray)),
                "contrast": float(np.std(gray)),
                "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "flags": [],
            }
        return {**detailed, **self._decision_flags(detailed)}

    def _decision_flags(self, data: Dict[str, object]) -> Dict[str, object]:
        brightness = float(data.get("brightness", 0.0))
        contrast = float(data.get("contrast", 0.0))
        blur_score = float(data.get("blur_score", 0.0))
        noise_score = float(data.get("noise_score", 0.0))
        return {
            "needs_denoise": noise_score > self.config.noise_threshold and blur_score > self.config.denoise_blur_threshold,
            "needs_contrast_boost": contrast < self.config.contrast_threshold or brightness < self.config.dark_threshold or brightness > self.config.bright_threshold,
            "needs_deskew": abs(float(data.get("skew_angle", 0.0))) > 0.35,
            "needs_shadow_correction": float(data.get("shadow_score", 0.0)) > 0.08,
            "needs_border_cleanup": float(data.get("border_score", 0.0)) > 0.08,
        }

    def _normalize_resolution(self, image: np.ndarray) -> Tuple[np.ndarray, str | None]:
        width = max(int(image.shape[1]), 1)
        if width < self.config.min_width:
            scale = self.config.min_width / width
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            return resized, f"upscale:{scale:.2f}"
        if width > self.config.max_width:
            scale = self.config.max_width / width
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            return resized, f"downscale:{scale:.2f}"
        return image, None

    def _perspective_correct(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, False

        image_area = float(gray.shape[0] * gray.shape[1])
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if area < image_area * 0.25:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            pts = approx.reshape(4, 2).astype(np.float32)
            warped = self._four_point_warp(image, pts)
            if warped.size:
                return warped, True
        return image, False

    def _four_point_warp(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        tl, tr, br, bl = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_width = int(max(width_a, width_b))
        max_height = int(max(height_a, height_b))
        if max_width < 100 or max_height < 100:
            return image
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height), borderMode=cv2.BORDER_REPLICATE)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        sums = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(sums)]
        rect[2] = pts[np.argmax(sums)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _correct_shadows(self, gray: np.ndarray) -> np.ndarray:
        kernel_size = max(31, int(min(gray.shape[:2]) * 0.06))
        if kernel_size % 2 == 0:
            kernel_size += 1
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        corrected = cv2.divide(gray, background, scale=245)
        return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _trim_dark_borders(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mask = gray > 35
        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(8, int(min(gray.shape[:2]) * 0.01))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        if (x2 - x1) < image.shape[1] * 0.70 or (y2 - y1) < image.shape[0] * 0.70:
            return image
        return image[y1:y2, x1:x2].copy()

    def _light_morphology(self, image: np.ndarray) -> np.ndarray:
        foreground_ratio = float(np.mean(image < 128))
        if foreground_ratio < 0.015 or foreground_ratio > 0.45:
            return image
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def _sauvola(self, gray: np.ndarray) -> np.ndarray | None:
        try:
            from skimage.filters import threshold_sauvola
        except Exception as exc:
            LOGGER.warning("sauvola threshold unavailable: %s", exc)
            return None
        window = max(25, int(min(gray.shape[:2]) * 0.035))
        if window % 2 == 0:
            window += 1
        threshold = threshold_sauvola(gray, window_size=window, k=0.2)
        return np.where(gray > threshold, 255, 0).astype(np.uint8)

    def _score_image(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        contrast = float(np.std(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        foreground_ratio = float(np.mean(gray < 210))
        contrast_score = min(1.0, contrast / 70.0)
        sharpness_score = min(1.0, blur / 450.0)
        brightness_score = 1.0 - min(1.0, abs(brightness - 172.0) / 172.0)
        foreground_score = 1.0 - min(1.0, abs(foreground_ratio - 0.18) / 0.35)
        return max(0.0, contrast_score * 0.32 + sharpness_score * 0.30 + brightness_score * 0.23 + foreground_score * 0.15)


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    border = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

