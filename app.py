from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Optional
from rag_engine import run_poc

app = FastAPI(title="GenAI Tender Evaluation", version="1.0.0")

# Response Models
class EvidenceQuote(BaseModel):
    citation_id: str
    quote: str

class EvaluationResult(BaseModel):
    criterion: str
    mandatory: bool
    compliant: bool
    confidence: str
    reasoning: str
    gaps: str

class BidderResult(BaseModel):
    bidder: str
    score: int
    mandatory_failed: bool
    evaluations: List[Dict]

class EvaluationResponse(BaseModel):
    winner: str
    ranked_bidders: List[BidderResult]

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    tender_file: UploadFile = File(..., description="The tender document (PDF)"),
    bidder_files: List[UploadFile] = File(..., description="Bidder proposals (PDF files)")
):
    """Evaluate tender bids and return rankings with detailed analysis."""
    
    # Validate inputs
    if not tender_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tender file is required"
        )
    
    if not tender_file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tender file must be a PDF"
        )
    
    if not bidder_files or len(bidder_files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one bidder file is required"
        )
    
    # Validate all bidder files are PDFs
    for bf in bidder_files:
        if not bf.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"All files must be PDFs. Invalid file: {bf.filename}"
            )
    
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # Save tender file
            tender_path = str(Path(tmp) / tender_file.filename)
            content = await tender_file.read()
            Path(tender_path).write_bytes(content)
            
            # Save bidder files
            bidders = {}
            for bf in bidder_files:
                filepath = str(Path(tmp) / bf.filename)
                content = await bf.read()
                Path(filepath).write_bytes(content)
                bidders[Path(bf.filename).stem] = filepath
            
            # Run evaluation
            result = run_poc(tender_path, bidders)
            return result
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
