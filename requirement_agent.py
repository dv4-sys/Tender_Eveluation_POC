from bedrock_llm import BedrockLLM
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_store import VectorStore

llm = BedrockLLM()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=12000,
    chunk_overlap=400,
    add_start_index=True,
    separators=["\n\n", "\n", "."],
)


def preprocess_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"â‚¹\s+(\d)", r"â‚¹\1", text)
    text = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_requirements(tender_text: str):
    cleaned_text = preprocess_text(tender_text)
    chunks = splitter.split_text(cleaned_text)
    if not chunks:
        return []

    chunk_docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_id": idx, "source": "tender"},
        )
        for idx, chunk in enumerate(chunks)
    ]

    tender_vs = VectorStore()
    tender_vs.build_index(chunk_docs)

    retrieval_queries = [
        "Eligibility Qualification requirements for bidders technical criteria",
        "Technical eligibility requirements SAP implementation AMC certified consultants authorized partner",
        "Financial eligibility requirements annual turnover working capital no default",
        "Bidder qualification conditions documentary evidence acceptance criteria",
    ]

    selected = {}
    k = 6 if len(chunk_docs) >= 6 else len(chunk_docs)
    for query in retrieval_queries:
        retrieved_docs = tender_vs.search(query, k=k)
        for doc in retrieved_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id not in selected:
                selected[chunk_id] = doc

    retrieved_context = "\n\n".join(
        [selected[key].page_content for key in sorted(selected.keys())]
    )

    prompt = f"""
You are a STRICT tender requirement extraction engine.

Task:
- Extract ONLY explicitly stated bidder eligibility/qualification requirements from Tender context "BID DATA SHEET".
- Include both Technical and Financial requirements.
- Do NOT infer missing details.
- Do NOT include non-eligibility operational clauses unless they are explicit bidder qualification criteria.

Return ONLY valid JSON array with this schema:
[
  {{
    "requirement_id": "REQ-001",
    "requirement_type": "short title",
    "category": "Technical | Financial",
    "description": "exact requirement text/description as stated in tender",
    "mandatory": true,
    "acceptance_criteria": "documentary proof required/evidence acceptance criteria as stated in tender"
  }}
]

Tender context (retrieved from semantic search):
{retrieved_context[:70000]}
"""

    result = llm.invoke_json(prompt)
    all_requirements = result if isinstance(result, list) else []

    normalized = []
    seen_keys = set()
    seq = 1

    for req in all_requirements:
        if not isinstance(req, dict):
            continue

        description = str(req.get("description", "")).strip()
        if not description:
            continue

        dedupe_key = " ".join(description.lower().split())
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        req_id = str(req.get("requirement_id", "")).strip()
        if not req_id:
            req_id = f"REQ-{seq:03d}"

        normalized.append(
            {
                "requirement_id": req_id,
                "requirement_type": str(req.get("requirement_type", "Requirement")).strip() or "Requirement",
                "category": str(req.get("category", "Technical")).strip() or "Technical",
                "description": description,
                "mandatory": bool(req.get("mandatory", True)),
                "acceptance_criteria": str(req.get("acceptance_criteria", "")).strip() or description,
            }
        )
        seq += 1

    return normalized
