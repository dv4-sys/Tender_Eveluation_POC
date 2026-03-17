# main.py
from fastapi import FastAPI, UploadFile, File
import shutil
import os
from pdf_parser import extract_text_from_pdf
from vector_store import VectorStore
from requirement_agent import extract_requirements
from compliance_agent import evaluate_requirement

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze/")
async def analyze(
    tender: UploadFile = File(...),
    bidders: list[UploadFile] = File(...)
):
    try:
        # Save tender
        tender_path = os.path.join(UPLOAD_DIR, tender.filename)
        with open(tender_path, "wb") as buffer:
            shutil.copyfileobj(tender.file, buffer)

        print(f"extract_text_from_pdf {tender.filename}")
        tender_docs = extract_text_from_pdf(tender_path, "tender")
        tender_text = "\n".join([d.page_content for d in tender_docs])

        print(f"extract_requirements")

        requirements = extract_requirements(tender_text)

        print(len(requirements))

        results = {}

        for bidder_file in bidders:
            bidder_path = os.path.join(UPLOAD_DIR, bidder_file.filename)
            with open(bidder_path, "wb") as buffer:
                shutil.copyfileobj(bidder_file.file, buffer)
            print(f"extract_text_from_pdf {bidder_file.filename}")
            bidder_docs = extract_text_from_pdf(bidder_path, "bidder")

            print("Creating VectorStore")
            vs = VectorStore()
            vs.build_index(bidder_docs)

            bidder_results = []

            for req in requirements:
                description = req.get("description", "")
                acceptance_criteria = req.get("acceptance_criteria", "")
                if not description:
                    continue

                retrieved = vs.search(description+acceptance_criteria, k=7)
                print("evaluate_requirement")
                evaluation = evaluate_requirement(req, retrieved, bidder_file.filename)
                requirement_id = (
                    req.get("requirement_id")
                    or "UNSPECIFIED"
                )
                evaluation_criterion = (
                    req.get("requirement_type")
                    or evaluation.get("evaluation_criterion")
                    or "UNSPECIFIED"
                )

                bidder_results.append({
                    "requirement_id": requirement_id,
                    "evaluation_criterion": evaluation_criterion,
                    "requirement_type": req.get("requirement_type", ""),
                    "description": description,
                    "acceptance_criteria": req.get("acceptance_criteria", ""),
                    "status": evaluation.get("status", "Not Found"),
                    "evidence_found": evaluation.get("evidence_found", []),
                    "gap_analysis": evaluation.get("gap_analysis", "")
                })

            results[bidder_file.filename] = bidder_results

        return {"results": results}
    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
