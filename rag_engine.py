# rag_engine.py
import os
import json
import csv
import hashlib
import datetime
import re
from pathlib import Path
from typing import List, Dict, Optional

import boto3
import pdfplumber
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from bedrock_llm import BedrockLLM

# -------------------------------------------------------------------
# ENV
# -------------------------------------------------------------------

load_dotenv()

boto3.setup_default_session(
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
llm = BedrockLLM()

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

VECTORSTORE_ROOT = Path("vectorstores")
VECTORSTORE_ROOT.mkdir(exist_ok=True)

DATA_DIR = Path("Data")
DATA_DIR.mkdir(exist_ok=True)

PDF_REGISTRY = DATA_DIR / "pdf_registry.csv"

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------

def log(msg, cb=None):
    print(msg)
    if cb:
        cb(msg)

# -------------------------------------------------------------------
# 🔥 ABSOLUTE KEY NORMALIZATION (FINAL FIX)
# -------------------------------------------------------------------

def normalize_keys(obj):
    """
    Recursively normalize JSON keys:
    - remove newlines
    - remove quotes
    - collapse spaces
    - lowercase
    """
    if isinstance(obj, dict):
        fixed = {}
        for k, v in obj.items():
            nk = str(k)
            nk = nk.replace("\n", "")
            nk = nk.replace('"', "")
            nk = nk.replace("'", "")
            nk = nk.strip()
            nk = re.sub(r"\s+", "", nk)
            nk = nk.lower()
            fixed[nk] = normalize_keys(v)
        return fixed
    elif isinstance(obj, list):
        return [normalize_keys(i) for i in obj]
    return obj

# -------------------------------------------------------------------
# JSON EXTRACTION
# -------------------------------------------------------------------

def extract_json_array(text: str) -> List[Dict]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found")
    return json.loads(text[start:end + 1])

def extract_json_object(text: str) -> Dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])

    raise ValueError("Unbalanced JSON object")

# -------------------------------------------------------------------
# PDF
# -------------------------------------------------------------------

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

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def chunk_documents(docs):
    out = []
    for d in docs:
        out.extend(splitter.split_documents([d]))
    return out

# -------------------------------------------------------------------
# ✅ CRITERIA EXTRACTION (IMPOSSIBLE TO FAIL)
# -------------------------------------------------------------------

def extract_tender_criteria(tender_text: str) -> List[Dict]:
    if not tender_text or not tender_text.strip():
        raise ValueError("Tender text is empty")

    # 1. Increase character limit to see more of the document
    max_chars = 15000 
    tender_text = tender_text[:max_chars]
    
    prompt = PromptTemplate(
        input_variables=["tender"],
        template="""Extract evaluation criteria from the tender document.

Return ONLY valid JSON array. Do NOT include explanations or markdown.

Expected format:
[
  {{
    "criterion_name": "string",
    "description": "string",
    "mandatory": boolean
  }}
]

Tender Text:
{tender}"""
    )

    response = llm.invoke(prompt.format(tender=tender_text))
    content = response.strip()

    try:
        criteria = extract_json_array(content)
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("Criteria must be a non-empty list")
        
        # FIX: Normalize the list of dictionaries before returning
        return normalize_keys(criteria) 
        
    except Exception as e:
        print(f"DEBUG: LLM returned: {content}")
        raise ValueError(f"Failed to parse criteria: {str(e)}")


# -------------------------------------------------------------------
# EVALUATION (CANNOT THROW KeyError)
# -------------------------------------------------------------------

def build_context(docs):
    ctx, cites = [], []
    for i, d in enumerate(docs, 1):
        cid = f"S{i}"
        ctx.append(f"[{cid}] {d.page_content}")
        cites.append({"id": cid, **d.metadata})
    return "\n\n".join(ctx), cites

def evaluate_criterion(criterion, store):
    docs = store.similarity_search(criterion["description"], k=5)

    if not docs:
        return {
            "compliant": False,
            "confidence": "Low",
            "reasoning": "No evidence found",
            "gaps": "Missing documentation",
            "score": 3,
            "citations": []
        }

    context, citations = build_context(docs)

    prompt = PromptTemplate(
    input_variables=["criterion", "context"],
    template="""
    Evaluate compliance.
    Return ONLY JSON object.
    {{
    "compliant": true | false,
    "confidence": "High" | "Medium" | "Low",
    "reasoning": "string",
    "gaps": "string"
    }}
    """
    )

    raw = llm.invoke(prompt.format(
        criterion=criterion["description"],
        context=context
    ))

    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = extract_json_object(raw)
        parsed = normalize_keys(parsed)
    except Exception:
        parsed = {}

    compliant = bool(parsed.get("compliant", False))

    return {
        "compliant": compliant,
        "confidence": parsed.get("confidence", "Low"),
        "reasoning": parsed.get("reasoning", "Unable to determine"),
        "gaps": parsed.get("gaps", ""),
        "score": 10 if compliant else 3,
        "citations": citations
    }

# -------------------------------------------------------------------
# BIDDER VALIDATION (SAFE)
# -------------------------------------------------------------------

def validate_bidder(name, criteria, store):
    total = 0
    failed = False
    evaluations = []

    for c in criteria:
        r = evaluate_criterion(c, store)
        total += r["score"]

        if c["mandatory"] and not r["compliant"]:
            failed = True

        evaluations.append({
            "criterion": c["criterion_name"],
            "mandatory": c["mandatory"],
            **r
        })

    score = min(100, int(total / (len(criteria) * 10) * 100))

    return {
        "bidder": name,
        "score": score,
        "mandatory_failed": failed,
        "evaluations": evaluations
    }

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def run_poc(tender_pdf: str, bidder_pdfs: Dict[str, str], progress_cb=None):
    log("Reading tender", progress_cb)
    tender_docs = extract_text_from_pdf(tender_pdf)

    log("Extracting criteria", progress_cb)
    criteria = extract_tender_criteria(
        "\n".join(d.page_content for d in tender_docs)
    )

    log("Creating vectorstore", progress_cb)
    all_docs = chunk_documents(tender_docs)

    for bidder, path in bidder_pdfs.items():
        docs = extract_text_from_pdf(path)
        for d in docs:
            d.metadata["source"] = bidder
        all_docs.extend(chunk_documents(docs))

    store = FAISS.from_documents(all_docs, embeddings)

    results = []
    for bidder in bidder_pdfs:
        log(f"Evaluating {bidder}", progress_cb)
        results.append(validate_bidder(bidder, criteria, store))

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    return {
        "winner": ranked[0]["bidder"],
        "ranked_bidders": ranked
    }
