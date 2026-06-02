"""Pydantic schema for invoice extraction results."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[str] = None
    unit_price: Optional[str] = None
    total: Optional[str] = None


class InvoiceSchema(BaseModel):
    document_type: str = "invoice"
    invoice_number: Optional[str] = None
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    seller_name: Optional[str] = None
    buyer_name: Optional[str] = None
    subtotal: Optional[str] = None
    vat: Optional[str] = None
    total: Optional[str] = None
    currency: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    raw_fields: Dict[str, Any] = Field(default_factory=dict)

