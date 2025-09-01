from typing import Dict, Any
import fitz  # PyMuPDF
import io


def parse_pdf(file) -> Dict[str, Any]:
    """
    Parse PDF file using PyMuPDF and return structured data.

    Args:
        file: File-like object (StreamlitUploadedFile or similar)

    Returns:
        Dict containing:
        - type: "pdf"
        - pages_text: List of text content for each page
        - text: Concatenated text from all pages
        - meta: PDF metadata
    """
    try:
        # Read the file content
        file_content = file.read()
        file.seek(0)  # Reset file pointer for potential future reads

        # Open PDF with PyMuPDF
        pdf_document = fitz.open(stream=file_content, filetype="pdf")

        pages_text = []
        all_text = ""

        # Extract text from each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            pages_text.append(page_text)
            all_text += page_text + "\n"

        # Extract metadata
        metadata = {}
        if pdf_document.metadata:
            metadata = {
                "title": pdf_document.metadata.get("title", ""),
                "author": pdf_document.metadata.get("author", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "creation_date": pdf_document.metadata.get("creationDate", ""),
                "modification_date": pdf_document.metadata.get("modDate", ""),
                "page_count": len(pdf_document),
            }

        pdf_document.close()

        return {
            "type": "pdf",
            "pages_text": pages_text,
            "text": all_text.strip(),
            "meta": metadata,
        }

    except Exception as e:
        # Return error information if parsing fails
        return {
            "type": "pdf",
            "pages_text": [],
            "text": f"Error parsing PDF: {str(e)}",
            "meta": {"error": str(e)},
        }
