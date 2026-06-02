import pytest
from pydantic import ValidationError
from app.ocr_pipeline.schemas.invoice import InvoiceSchema, LineItem
from app.ocr_pipeline.schemas.cv import CVSchema

def test_valid_invoice_validation():
    data = {
        "invoice_number": "INV-001",
        "total": "150.50",
        "line_items": [
            {"description": "Item 1", "total": "50.50"},
            {"description": "Item 2", "total": "100.00"}
        ]
    }
    invoice = InvoiceSchema(**data)
    assert invoice.invoice_number == "INV-001"
    assert len(invoice.line_items) == 2

def test_valid_cv_validation():
    data = {
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "experience": [
            {"company": "Google", "title": "Software Engineer"}
        ]
    }
    cv = CVSchema(**data)
    assert cv.full_name == "Jane Doe"
    assert cv.email == "jane@example.com"

def test_missing_required_fields():
    # In our schemas, most fields are Optional, but document_type has a default.
    # Let's see if we can force an error by passing wrong types if any.
    with pytest.raises(ValidationError):
        # line_items should be a list
        InvoiceSchema(line_items="not a list")
