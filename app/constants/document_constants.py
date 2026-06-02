"""
app/constants/document_constants.py – Constants for document upload
"""

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpeg", ".jpg", ".tiff"}

MAX_FILE_SIZE = 5 * 1024 * 1024

MAGIC_BYTES = {
    ".pdf": [b"%PDF"],
    ".png": [b"\x89PNG"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".jpg": [b"\xff\xd8\xff"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
}