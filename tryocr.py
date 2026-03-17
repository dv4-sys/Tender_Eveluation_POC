import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\dv4\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

pdf_path = r"uploads\YASH Technologies.pdf"

print(f"Processing PDF: {pdf_path}")

doc = fitz.open(pdf_path)

text = []

for page_num, page in enumerate(doc):

    # 1️⃣ Try native text extraction
    native_text = page.get_text("text")

    if native_text.strip() and len(native_text.strip()) > 50:
        print(f"Page {page_num+1}: native text extracted")
        text.append(native_text)
        continue

    # 2️⃣ Run OCR only if needed
    print(f"Page {page_num+1}: running OCR")

    pix = page.get_pixmap(dpi=200)  # lower DPI = faster

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    ocr_text = pytesseract.image_to_string(
        img,
        lang="eng",
        config="--oem 3 --psm 6"
    )

    text.append(ocr_text)

doc.close()

final_text = "\n".join(text)

print(final_text)