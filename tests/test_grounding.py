import pytest
from app.ocr_pipeline.validation.grounding import GroundingChecker

@pytest.fixture
def checker():
    return GroundingChecker()

def test_grounding_accepted(checker):
    extracted = {"invoice_number": "INV-123", "total": "100.00"}
    source_text = "Invoice INV-123 Total: 100.00 USD"
    
    result = checker.verify(extracted, source_text)
    assert result["is_grounded"] is True
    assert result["field_status"]["invoice_number"] is True
    assert result["field_status"]["total"] is True

def test_grounding_rejected(checker):
    extracted = {"invoice_number": "INV-999"} # Not in source
    source_text = "Invoice INV-123 Total: 100.00 USD"
    
    result = checker.verify(extracted, source_text)
    assert result["is_grounded"] is False
    assert result["field_status"]["invoice_number"] is False

def test_grounding_partial_match(checker):
    # Test that partial matches or minor noise are handled (if fuzzy_check is used)
    extracted = {"company": "Tech Solutions Inc"}
    source_text = "Tech Solutions Incorporated"
    
    result = checker.verify(extracted, source_text)
    # Our simple fuzzy check looks at first/last 10 chars
    # "Tech Solut" is in "Tech Solutions Incorporated"
    assert result["field_status"]["company"] is True

def test_grounding_case_insensitivity(checker):
    extracted = {"name": "JOHN DOE"}
    source_text = "john doe was here"
    
    result = checker.verify(extracted, source_text)
    assert result["field_status"]["name"] is True
