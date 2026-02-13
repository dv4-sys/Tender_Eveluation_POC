from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import tempfile
from pathlib import Path
from typing import List, Dict
# from rag_engine import run_poc
from rag.pipeline import run_poc


app = FastAPI(title="GenAI Tender Evaluation", version="1.0.0")

# ------------------------------------------------------------------
# RESPONSE MODELS (match rag_engine output)
# ------------------------------------------------------------------

class EvaluationItem(BaseModel):
    criterion: str
    mandatory: bool
    compliant: bool
    confidence: str
    reasoning: str
    gaps: str
    score: int
    citations: List[Dict]

class BidderResult(BaseModel):
    bidder: str
    score: int
    mandatory_failed: bool
    evaluations: List[EvaluationItem]

class EvaluationResponse(BaseModel):
    winner: str | None
    ranked_bidders: List[BidderResult]

# ------------------------------------------------------------------
# API ENDPOINT
# ------------------------------------------------------------------

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    tender_file: UploadFile = File(..., description="Tender document (PDF)"),
    bidder_files: List[UploadFile] = File(..., description="Bidder proposal PDFs"),
):
    """
    Evaluate bidder proposals against a tender document
    using RAG + Amazon Bedrock.
    """

    # -------------------- VALIDATION --------------------

    if not tender_file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tender file must be a PDF",
        )

    if not bidder_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one bidder file is required",
        )

    for bf in bidder_files:
        if not bf.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid bidder file (not PDF): {bf.filename}",
            )

    # -------------------- PROCESSING --------------------

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save tender
            tender_path = tmp_path / tender_file.filename
            tender_path.write_bytes(await tender_file.read())

            # Save bidders
            bidders: Dict[str, str] = {}
            for bf in bidder_files:
                file_path = tmp_path / bf.filename
                file_path.write_bytes(await bf.read())
                bidders[file_path.stem] = str(file_path)

            # Run core evaluation
            result = run_poc(str(tender_path), bidders)

            return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {e}",
        )

# ------------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
