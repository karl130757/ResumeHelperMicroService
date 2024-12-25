from pdfminer.high_level import extract_text
import docx
import os

def parse_pdf(file_path):
    return extract_text(file_path)

def parse_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def parse_file(file_path):
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    else:
        raise ValueError("Unsupported file type")
