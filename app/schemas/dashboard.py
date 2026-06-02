""" 
app/schemas/dashboard.py - Pydantic schema for dashboard statistics
"""

from pydantic import BaseModel

class DashboardStats(BaseModel):
    """ 
    Statistics displayed in the admin dashboard
    Calculated from the entire database
    """
    documents_processed: int # total number of uploaded documents
    extraction_success_rate: float # percentage of documents with confidence_score >= 0.50
    low_confidence_documents: int # number of documents with confidence_score < 0.50
    most_processed_doc_type: str | None 
    # most frequent document type detected by the pipeline
    # None if no document has been processed yet

