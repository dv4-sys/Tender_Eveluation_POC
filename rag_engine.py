import os
import json
import hashlib
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# LLM Setup (singleton pattern to reuse across invocations)
llm = ChatOpenAI(model="gpt-4o", temperature=0, request_timeout=60)
embeddings = OpenAIEmbeddings()

# Persistent vector store base directory
VECTORSTORE_ROOT = Path("vectorstores")
VECTORSTORE_ROOT.mkdir(exist_ok=True)


def _make_store_id(tender_pdf: str, bidder_pdfs: Dict[str, str]) -> str:
    """Create a stable id based on file paths and mtimes to detect changes."""
    m = hashlib.md5()
    def fed(path: str):
        try:
            stat = os.path.getmtime(path)
        except Exception:
            stat = 0
        m.update(f"{path}:{stat}".encode("utf-8"))

    fed(tender_pdf)
    for k in sorted(bidder_pdfs.keys()):
        fed(bidder_pdfs[k])

    return m.hexdigest()


def _load_vector_store_if_exists(store_id: str):
    path = VECTORSTORE_ROOT / store_id
    if not path.exists():
        return None
    try:
        return FAISS.load_local(str(path), embeddings)
    except Exception as e:
        print(f"Warning: failed to load vector store {path}: {e}")
        return None


def _save_vector_store(store, store_id: str):
    path = VECTORSTORE_ROOT / store_id
    path.mkdir(parents=True, exist_ok=True)
    try:
        store.save_local(str(path))
    except Exception as e:
        print(f"Warning: failed to save vector store {path}: {e}")

# PDF Text Extraction (Page-Aware) with error handling
def extract_text_from_pdf(path: str) -> List[Document]:
    """Extract text from PDF with page metadata. Optimized for large documents."""
    docs = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": os.path.basename(path),
                                    "page": i + 1
                                }
                            )
                        )
                except Exception as e:
                    print(f"Warning: Failed to extract page {i + 1}: {str(e)}")
                    continue
    except Exception as e:
        raise ValueError(f"Failed to process PDF {path}: {str(e)}")
    
    if not docs:
        raise ValueError(f"No text content found in PDF: {path}")
    
    return docs


# Chunking (singleton splitter to avoid recreation)
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks for efficient embedding."""
    if not docs:
        return []
    chunks: List[Document] = []
    # Split each source document individually so we can retain per-page metadata
    for doc in docs:
        # split_documents accepts a list; pass single-doc to keep mapping clear
        parts = _splitter.split_documents([doc])
        for part in parts:
            # Ensure metadata preserves original source/page information
            base_meta = dict(doc.metadata) if getattr(doc, "metadata", None) else {}
            part_meta = dict(getattr(part, "metadata", {}) or {})
            # Merge, preferring any metadata already present on the split part
            merged = {**base_meta, **part_meta}
            part.metadata = merged
            chunks.append(part)

    return chunks


# Tender Criteria Extraction (SAFE JSON)
def extract_tender_criteria(tender_text: str) -> List[Dict]:
    """Extract and parse evaluation criteria from tender document."""
    if not tender_text or not tender_text.strip():
        raise ValueError("Tender text is empty")
    
    # Limit text to avoid token overflow
    max_chars = 4000
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
    content = response.content.strip()
    
    # Clean common LLM wrappers
    content = content.replace("```json", "").replace("```", "").strip()
    
    try:
        criteria = json.loads(content)
        # Validate structure
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("Criteria must be a non-empty list")
        return criteria
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse criteria JSON: {str(e)}. Response: {content[:500]}")



# Context Builder with Citations
def build_context(docs):
    context = []
    citations = []

    for i, d in enumerate(docs, start=1):
        cid = f"S{i}"
        context.append(
            f"[{cid}] ({d.metadata['source']} | Page {d.metadata['page']})\n{d.page_content}"
        )
        citations.append({
            "citation_id": cid,
            "source": d.metadata["source"],
            "page": d.metadata["page"],
            "snippet": d.page_content[:300]
        })

    return "\n\n".join(context), citations


# Criterion Validation (Explainable)
def evaluate_criterion(criterion: Dict, vector_store) -> Dict:
    """Evaluate a single criterion against vector store evidence."""
    try:
        # Search for most relevant documents
        docs = vector_store.similarity_search(criterion["description"], k=5)
        
        if not docs:
            return {
                "compliant": False,
                "confidence": "Low",
                "reasoning": "No relevant evidence found in documents",
                "evidence_quotes": [],
                "gaps": "No supporting documentation provided",
                "citations": []
            }
        
        context, citations = build_context(docs)
        
        prompt = PromptTemplate(
            input_variables=["criterion", "context"],
            template="""Evaluate if the following requirement is met.

Criterion:
{criterion}

Evidence:
{context}

Return ONLY valid JSON:
{{
  "compliant": boolean,
  "confidence": "High|Medium|Low",
  "reasoning": "short explanation",
  "evidence_quotes": [
    {{
      "citation_id": "S1",
      "quote": "exact text"
    }}
  ],
  "gaps": "missing items or empty string"
}}"""
        )
        
        response = llm.invoke(
            prompt.format(
                criterion=criterion["description"],
                context=context
            )
        )
        
        result = response.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(result)
        parsed["citations"] = citations

        # Compute a partial score (0-10) for this criterion to allow partial credit
        try:
            compliant = bool(parsed.get("compliant", False))
            confidence = str(parsed.get("confidence", "Low")).lower()
            gaps = str(parsed.get("gaps", "") or "").strip()

            if compliant:
                score = 10
            else:
                # Give partial credit based on confidence when not fully compliant
                if confidence.startswith("high"):
                    score = 7
                elif confidence.startswith("medium"):
                    score = 4
                elif confidence.startswith("low"):
                    score = 1
                else:
                    score = 2

            # Penalize for explicitly listed gaps
            if gaps:
                try:
                    score = max(0, score - 2)
                except Exception:
                    pass

            parsed["score"] = int(score)
        except Exception:
            parsed["score"] = 0

        return parsed
    except Exception as e:
        return {
            "compliant": False,
            "confidence": "Low",
            "reasoning": f"Evaluation error: {str(e)}",
            "evidence_quotes": [],
            "gaps": "Unable to evaluate",
            "citations": []
        }



# Bidder Validation
def validate_bidder(bidder_name: str, criteria: List[Dict], vector_store) -> Dict:
    """Evaluate a bidder against all criteria and compute confidence-weighted scores."""
    evaluations = []
    raw_score = 0
    weighted_score = 0.0
    mandatory_failed = False

    def _confidence_factor(confidence: Optional[str]) -> float:
        if not confidence:
            return 0.6
        c = str(confidence).lower()
        if c.startswith("high"):
            return 1.0
        if c.startswith("medium"):
            return 0.8
        if c.startswith("low"):
            return 0.5
        return 0.6

    for criterion in criteria:
        result = evaluate_criterion(criterion, vector_store)

        crit_score = int(result.get("score", 0))
        raw_score += crit_score

        factor = _confidence_factor(result.get("confidence"))
        weighted_score += crit_score * factor

        if not result.get("compliant", False) and criterion.get("mandatory", False):
            mandatory_failed = True

        evaluations.append({
            "criterion": criterion.get("criterion_name", "Unknown"),
            "mandatory": criterion.get("mandatory", False),
            **result
        })

    max_raw = len(criteria) * 10 if criteria else 1
    raw_pct = int(min(100, (raw_score / max_raw) * 100))
    weighted_pct = int(min(100, (weighted_score / max_raw) * 100))

    return {
        "bidder": bidder_name,
        "raw_score": raw_pct,
        "weighted_score": weighted_pct,
        "score": weighted_pct,
        "mandatory_failed": mandatory_failed,
        "evaluations": evaluations
    }


# MAIN ENTRYPOINT
def run_poc(tender_pdf: str, bidder_pdfs: Dict[str, str]) -> Dict:
    """Main evaluation engine: extract criteria, evaluate bidders, return rankings."""
    if not tender_pdf or not os.path.exists(tender_pdf):
        raise ValueError(f"Invalid tender PDF path: {tender_pdf}")
    
    if not bidder_pdfs:
        raise ValueError("No bidder documents provided.")
    
    # Validate all bidder paths exist
    for bidder_name, path in bidder_pdfs.items():
        if not os.path.exists(path):
            raise ValueError(f"Bidder file not found: {path}")
    
    print("[1/4] Extracting tender document...")
    tender_docs = extract_text_from_pdf(tender_pdf)
    tender_text = "\n".join(d.page_content for d in tender_docs)
    
    print("[2/4] Extracting evaluation criteria...")
    criteria = extract_tender_criteria(tender_text)
    
    if not criteria:
        raise ValueError("No evaluation criteria could be extracted from tender.")
    
    print(f"[3/4] Building vector store with {len(bidder_pdfs)} bidders...")
    # Chunk tender documents
    all_docs = chunk_documents(tender_docs)
    
    # Process bidder documents
    for bidder_name, bidder_path in bidder_pdfs.items():
        try:
            bidder_docs = extract_text_from_pdf(bidder_path)
            # Tag bidder source
            for doc in bidder_docs:
                doc.metadata["source"] = bidder_name
            all_docs.extend(chunk_documents(bidder_docs))
        except Exception as e:
            print(f"Warning: Failed to process bidder '{bidder_name}': {str(e)}")
    
    if not all_docs:
        raise ValueError("No valid documents to process.")
    
    # Create or load persistent vector store (avoids recomputing embeddings)
    store_id = _make_store_id(tender_pdf, bidder_pdfs)
    vector_store = _load_vector_store_if_exists(store_id)
    if vector_store is None:
        vector_store = FAISS.from_documents(all_docs, embeddings)
        _save_vector_store(vector_store, store_id)
    else:
        print(f"Loaded vector store: {store_id}")
    
    print("[4/4] Evaluating bidders...")
    results = []
    for bidder_name in bidder_pdfs:
        try:
            res = validate_bidder(bidder_name, criteria, vector_store)
            results.append(res)
        except Exception as e:
            print(f"Warning: evaluation failed for bidder {bidder_name}: {e}")

    if not results:
        return {
            "winner": None,
            "ranked_bidders": [],
            "error": "No bidders could be evaluated."
        }

    # Rank bidders by weighted score (confidence-weighted)
    ranked = sorted(results, key=lambda x: x.get("weighted_score", x.get("score", 0)), reverse=True)

    return {
        "winner": ranked[0]["bidder"],
        "ranked_bidders": ranked
    }
