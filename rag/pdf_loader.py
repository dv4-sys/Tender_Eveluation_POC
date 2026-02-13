import os
import pdfplumber
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document

def extract_text_from_pdf(path: str) -> List[Document]:
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            t = page.extract_text()
            if t:
                docs.append(Document(
                    page_content=t,
                    metadata={"source": os.path.basename(path), "page": i + 1}
                ))
    if not docs:
        raise ValueError(f"No text in {path}")
    return docs

def extract_many_pdfs(pdf_map: Dict[str, str]) -> Dict[str, List[Document]]:
    with ThreadPoolExecutor(max_workers=4) as ex:
        return dict(zip(
            pdf_map.keys(),
            ex.map(extract_text_from_pdf, pdf_map.values())
        ))
