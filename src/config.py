import os

class Config:
    UPLOAD_FOLDER = "uploads"
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
