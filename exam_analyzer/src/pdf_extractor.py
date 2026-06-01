"""PDF text + tables extraction. pdfplumber primary, PyMuPDF fallback."""

import os
import pdfplumber

from .models import ExtractedPage, ExtractedPDF


def _extract_with_pdfplumber(pdf_path: str) -> ExtractedPDF:
    """Extract with pdfplumber. Failed pages get targeted PyMuPDF fallback.
    If >30% of pages fail, falls back entirely to PyMuPDF."""
    pages: list[ExtractedPage] = []
    all_text_parts = []
    failed_pages: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
                raw_tables = page.extract_tables()
                tables: list[list[list[str]]] = []
                if raw_tables:
                    for t in raw_tables:
                        cleaned = [
                            [cell if cell else "" for cell in row]
                            for row in t
                        ]
                        tables.append(cleaned)
                page_text = text.strip()
            except Exception as e:
                from .error_utils import log_exception
                log_exception(print, "PDF page", f"page={i}", e)
                page_text = ""
                tables = []
                failed_pages.append(i)

            pages.append(ExtractedPage(text=page_text, tables=tables))
            all_text_parts.append(page_text)

    # If too many pages failed, fall back entirely to PyMuPDF
    if len(failed_pages) > len(pages) * 0.3:
        return _extract_with_pymupdf(pdf_path)

    # Targeted PyMuPDF fallback for individual failed pages
    if failed_pages:
        import fitz
        with fitz.open(pdf_path) as doc:
            for i in failed_pages:
                if i < len(doc):
                    fb_text = doc[i].get_text("text")
                    if not fb_text.strip():
                        blocks = doc[i].get_text("blocks")
                        fb_text = "\n".join(
                            b[4] for b in blocks
                            if isinstance(b, (list, tuple)) and len(b) >= 5 and isinstance(b[4], str)
                        )
                    pages[i] = ExtractedPage(text=fb_text.strip(), tables=[])
                    all_text_parts[i] = fb_text.strip()

    full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text_parts)
    return ExtractedPDF(pages=pages, full_text=full_text, used_fallback=bool(failed_pages))


def _fallback_pymupdf(pdf_path: str) -> str:
    """PyMuPDF text extraction."""
    import fitz
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if not text.strip():
                blocks = page.get_text("blocks")
                text = "\n".join(b[4] for b in blocks if isinstance(b, (list, tuple)) and len(b) >= 5 and isinstance(b[4], str))
            pages.append(text if text.strip() else "")
    return "\n\n--- PAGE BREAK ---\n\n".join(pages)


def _extract_with_pymupdf(pdf_path: str) -> ExtractedPDF:
    """Fallback extraction using PyMuPDF (no tables)."""
    full_text = _fallback_pymupdf(pdf_path)

    page_texts = full_text.split("\n\n--- PAGE BREAK ---\n\n")
    pages = [ExtractedPage(text=t.strip()) for t in page_texts if t.strip()]

    return ExtractedPDF(pages=pages, full_text=full_text, used_fallback=True)


def extract_pdf(pdf_path: str) -> ExtractedPDF:
    """
    Extract text + tables from a PDF.
    Primary: pdfplumber. Fallback: PyMuPDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Try pdfplumber first
    try:
        return _extract_with_pdfplumber(pdf_path)
    except Exception as e:
        from .error_utils import log_exception
        log_exception(print, "PDF extract", f"file={os.path.basename(pdf_path)}", e, level="warning")
        print(f"  Falling back to PyMuPDF...")
        return _extract_with_pymupdf(pdf_path)
