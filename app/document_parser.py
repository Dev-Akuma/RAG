import io
import fitz  # PyMuPDF
from docx import Document
import email
from email import policy
import extract_msg

SUPPORTED_FORMATS = [".pdf", ".docx", ".eml", ".msg"]

def extract_text_from_pdf_file(file_stream) -> str:
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = []
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}")

def extract_text_from_docx_file(file_stream) -> str:
    try:
        # Need to reset stream position for python-docx
        file_stream.seek(0)
        doc = Document(file_stream)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        raise RuntimeError(f"Failed to parse DOCX: {e}")

def extract_text_from_eml_file(file_stream) -> str:
    try:
        file_stream.seek(0)
        msg = email.message_from_binary_file(file_stream, policy=policy.default)
        subject = msg.get("subject", "")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_content()
        else:
            body = msg.get_content()
        return f"Subject: {subject}\n\n{body.strip()}"
    except Exception as e:
        raise RuntimeError(f"Failed to parse EML: {e}")

def extract_text_from_msg_file(file_stream) -> str:
    try:
        # extract_msg requires a file path, so we'll save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(file_stream.read())
            tmp.flush()
            msg = extract_msg.Message(tmp.name)
            return f"Subject: {msg.subject}\n\n{msg.body.strip()}"
    except Exception as e:
        raise RuntimeError(f"Failed to parse MSG: {e}")

def extract_text_from_upload(file_stream, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    ext = "." + ext

    if ext == ".pdf":
        return extract_text_from_pdf_file(file_stream)
    elif ext == ".docx":
        return extract_text_from_docx_file(file_stream)
    elif ext == ".eml":
        return extract_text_from_eml_file(file_stream)
    elif ext == ".msg":
        return extract_text_from_msg_file(file_stream)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {SUPPORTED_FORMATS}")
