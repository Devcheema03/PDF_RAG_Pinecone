import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF.

    Normal PDFs are read using PyMuPDF.
    Scanned/image pages are processed using Tesseract OCR.
    """

    document = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(document, start=1):

        # Try normal PDF text extraction first
        text = page.get_text("text").strip()

        # If normal text is available, use it
        if text:
            pages.append({
                "page_number": page_number,
                "text": text
            })

        # Otherwise, use OCR
        else:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            image_bytes = pixmap.tobytes("png")
            image = Image.open(io.BytesIO(image_bytes))

            ocr_text = pytesseract.image_to_string(image).strip()

            if ocr_text:
                pages.append({
                    "page_number": page_number,
                    "text": ocr_text
                })

    document.close()

    return pages