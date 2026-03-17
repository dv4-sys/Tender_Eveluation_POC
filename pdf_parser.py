import os
import fitz
import pytesseract
from PIL import Image
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document

# ----------------------------------------
# Utility: Extract Text from PDF
# ----------------------------------------

def extract_text_from_pdf(
    path: str,
    doc_type: str = "bidder",
    *,
    use_ocr_when_needed: bool = True,
    ocr_dpi: int = 200,
    ocr_language: str = "eng",
    tesseract_cmd: Optional[str] = r"C:\Users\dv4\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
) -> List[Document]:

    print("Running extraction (native + pytesseract fallback)...")

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    docs: List[Document] = []
    pdf_doc = fitz.open(path)

    try:
        for page in pdf_doc:
            page_number = page.number + 1

            native_text = page.get_text("text")
            text_length = len(native_text.strip()) if native_text else 0

            # Detect images in page
            image_count = len(page.get_images(full=True))

            # Heuristic: if enough text and no large images → native text
            if text_length > 100 and image_count == 0:
                docs.append(
                    Document(
                        page_content=native_text,
                        metadata={
                            "source": os.path.basename(path),
                            "page": page_number,
                            "doc_type": doc_type,
                            "extraction_method": "pymupdf-native",
                        },
                    )
                )
                continue

            # OCR fallback
            if use_ocr_when_needed:
                print(f"OCR running on page {page_number}...")

                pix = page.get_pixmap(dpi=ocr_dpi)

                img = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples,
                )

                ocr_text = pytesseract.image_to_string(
                    img,
                    lang=ocr_language,
                    config="--oem 3 --psm 6",
                )

                if ocr_text.strip():
                    docs.append(
                        Document(
                            page_content=ocr_text,
                            metadata={
                                "source": os.path.basename(path),
                                "page": page_number,
                                "doc_type": doc_type,
                                "extraction_method": "pytesseract-ocr",
                                "ocr_language": ocr_language,
                                "ocr_dpi": ocr_dpi,
                            },
                        )
                    )

        if not docs:
            raise ValueError(f"No text extracted from {path}")

        os.makedirs("ocr", exist_ok=True)

        log_path = f"ocr/{os.path.basename(path)}_extraction_log.txt"

        with open(log_path, "w", encoding="utf-8") as log_file:
            for extracted_doc in docs:
                log_file.write(
                    f"Page {extracted_doc.metadata.get('page')} "
                    f"({extracted_doc.metadata.get('extraction_method')}):\n"
                )
                log_file.write(extracted_doc.page_content)
                log_file.write("\n\n---\n\n")

        print(f"Extraction log saved to: {log_path}")

        return docs

    finally:
        pdf_doc.close()




def extract_many_pdfs(pdf_map: Dict[str, str], cached_vectorstore_ids: Dict[str, str] = None) -> Dict[str, List[Document]]:
    """
    Extract text from multiple PDFs concurrently.
    If cached_vectorstore_ids is provided, skip extraction for those PDFs.
    Returns only the documents for PDFs that need extraction (not found in cache).
    """
    if cached_vectorstore_ids is None:
        cached_vectorstore_ids = {}
    
    # Filter out PDFs that are already cached
    pdfs_to_extract = {
        name: path for name, path in pdf_map.items() 
        if name not in cached_vectorstore_ids
    }
    
    if not pdfs_to_extract:
        print("All bidders found in cache")
        return {}
    
    with ThreadPoolExecutor(max_workers=4) as ex:
        return dict(
            zip(
                pdfs_to_extract.keys(),
                ex.map(
                    lambda p: extract_text_from_pdf(p, doc_type="bidder"),
                    pdfs_to_extract.values(),
                ),
            )
        )

