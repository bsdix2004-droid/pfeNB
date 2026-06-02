"""Image quality analysis used by the adaptive preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass(slots=True)
class ImageQualityProfile:
    """Compact quality profile for a document image."""

    width: int
    height: int
    channels: int
    brightness: float
    contrast: float
    blur_score: float
    noise_score: float
    skew_angle: float
    shadow_score: float
    border_score: float
    document_area_ratio: float
    perspective_confidence: float
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "brightness": round(self.brightness, 3),
            "contrast": round(self.contrast, 3),
            "blur_score": round(self.blur_score, 3),
            "noise_score": round(self.noise_score, 3),
            "skew_angle": round(self.skew_angle, 3),
            "shadow_score": round(self.shadow_score, 3),
            "border_score": round(self.border_score, 3),
            "document_area_ratio": round(self.document_area_ratio, 4),
            "perspective_confidence": round(self.perspective_confidence, 3),
            "flags": list(self.flags),
        }


class ImageQualityAnalyzer:
    """Estimates OCR-relevant image quality without running OCR."""

    def analyze(self, image: np.ndarray) -> ImageQualityProfile:
        if image is None or image.size == 0:
            raise ValueError("Cannot analyze an empty image")

        gray = self._to_gray(image)
        height, width = gray.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise_score = self._estimate_noise(gray)
        skew_angle = self.estimate_skew_angle(gray)
        shadow_score = self._estimate_shadow(gray)
        border_score = self._estimate_border(gray)
        area_ratio, perspective_conf = self._estimate_document_contour(gray)

        flags = self._build_flags(
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
            noise_score=noise_score,
            skew_angle=skew_angle,
            shadow_score=shadow_score,
            border_score=border_score,
            perspective_confidence=perspective_conf,
        )

        return ImageQualityProfile(
            width=width,
            height=height,
            channels=channels,
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
            noise_score=noise_score,
            skew_angle=skew_angle,
            shadow_score=shadow_score,
            border_score=border_score,
            document_area_ratio=area_ratio,
            perspective_confidence=perspective_conf,
            flags=flags,
        )

    def estimate_skew_angle(self, image: np.ndarray) -> float:
        """Estimate dominant text/document skew in degrees."""
        gray = self._to_gray(image)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        min_line_length = max(40, int(min(gray.shape[:2]) * 0.15))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min_line_length,
            maxLineGap=20,
        )

        angles: List[float] = []
        if lines is not None:
            for item in lines[:, 0]:
                x1, y1, x2, y2 = [int(v) for v in item]
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) < 8:
                    continue
                angle = float(np.degrees(np.arctan2(dy, dx)))
                if -30.0 <= angle <= 30.0:
                    angles.append(angle)

        if angles:
            return float(np.median(angles))

        # Fallback: infer skew from foreground pixels after adaptive binarization.
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            15,
        )
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 100:
            return 0.0
        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = float(rect[-1])
        if angle < -45:
            angle = 90 + angle
        if angle > 45:
            angle = angle - 90
        return angle if -30 <= angle <= 30 else 0.0

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _estimate_noise(self, gray: np.ndarray) -> float:
        denoised = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, denoised)
        return float(np.std(residual))

    def _estimate_shadow(self, gray: np.ndarray) -> float:
        kernel = max(31, int(min(gray.shape[:2]) * 0.08))
        if kernel % 2 == 0:
            kernel += 1
        background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        return float(np.std(background) / 255.0)

    def _estimate_border(self, gray: np.ndarray) -> float:
        h, w = gray.shape[:2]
        margin_y = max(2, int(h * 0.025))
        margin_x = max(2, int(w * 0.025))
        strips = [
            gray[:margin_y, :],
            gray[h - margin_y :, :],
            gray[:, :margin_x],
            gray[:, w - margin_x :],
        ]
        dark_ratios = [float(np.mean(strip < 35)) for strip in strips if strip.size]
        return float(np.mean(dark_ratios)) if dark_ratios else 0.0

    def _estimate_document_contour(self, gray: np.ndarray) -> Tuple[float, float]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0

        image_area = float(gray.shape[0] * gray.shape[1])
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        area_ratio = area / max(image_area, 1.0)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        confidence = 0.0
        if len(approx) == 4 and 0.25 <= area_ratio <= 0.98:
            confidence = min(1.0, area_ratio * 1.2)
        elif 0.45 <= area_ratio <= 0.98:
            confidence = 0.35
        return area_ratio, confidence

    def _build_flags(self, **values: float) -> List[str]:
        flags: List[str] = []
        if values["brightness"] < 85:
            flags.append("dark")
        if values["brightness"] > 210:
            flags.append("bright")
        if values["contrast"] < 38:
            flags.append("low_contrast")
        if values["blur_score"] < 90:
            flags.append("blurred")
        if values["noise_score"] > 9:
            flags.append("noisy")
        if abs(values["skew_angle"]) > 0.4:
            flags.append("skewed")
        if values["shadow_score"] > 0.10:
            flags.append("uneven_lighting")
        if values["border_score"] > 0.15:
            flags.append("dark_borders")
        if values["perspective_confidence"] > 0.55:
            flags.append("document_boundary_detected")
        return flags

