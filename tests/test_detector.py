import pytest
from unittest.mock import MagicMock
from app.ocr_pipeline.core.detector import DocumentDetector, DocumentInfo
from app.ocr_pipeline.classifiers.document_classifier import ClassificationResult

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.OCR_PIPELINE.language.use_fasttext = False
    settings.OCR_PIPELINE.language.use_cld3 = False
    settings.OCR_PIPELINE.language.supported_languages = ["en", "fr", "ar"]
    settings.OCR_PIPELINE.detector.min_confidence = 0.20
    return settings

@pytest.fixture
def detector(mock_settings):
    # Mock DocumentClassifier and LanguageIdentifier to avoid loading models
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.ocr_pipeline.core.detector.DocumentClassifier", MagicMock())
        mp.setattr("app.ocr_pipeline.core.detector.LanguageIdentifier", MagicMock())
        return DocumentDetector(mock_settings)

def test_detect_invoice(detector):
    # Mock classification result
    detector.classifier.classify.return_value = ClassificationResult(
        doc_type="invoice",
        confidence=0.95,
        scores={"invoice": 0.95, "cv": 0.05},
        evidence={"keywords": ["invoice", "total"]}
    )
    detector.language_identifier.identify.return_value = MagicMock(
        primary_language="en",
        direction="ltr",
        probabilities={"en": 1.0}
    )

    result = detector.detect(None, "This is an invoice with total $100")
    
    assert result.doc_type == "invoice"
    assert result.confidence == 0.95
    assert result.language == "en"

def test_detect_cv(detector):
    detector.classifier.classify.return_value = ClassificationResult(
        doc_type="cv",
        confidence=0.88,
        scores={"cv": 0.88, "invoice": 0.02},
        evidence={"keywords": ["experience", "education"]}
    )
    detector.language_identifier.identify.return_value = MagicMock(
        primary_language="en",
        direction="ltr",
        probabilities={"en": 1.0}
    )

    result = detector.detect(None, "John Doe Resume Experience: Python Developer")
    
    assert result.doc_type == "cv"
    assert result.confidence == 0.88

def test_detect_unknown_fallback(detector):
    # Low confidence should fallback to unknown
    detector.classifier.classify.return_value = ClassificationResult(
        doc_type="invoice",
        confidence=0.15, # Below threshold
        scores={"invoice": 0.15},
        evidence={}
    )
    detector.language_identifier.identify.return_value = MagicMock(
        primary_language="en",
        direction="ltr",
        probabilities={"en": 1.0}
    )

    result = detector.detect(None, "Some random text")
    
    # Now it should return "unknown" due to low confidence
    assert result.doc_type == "unknown"
    assert result.confidence == 0.15
