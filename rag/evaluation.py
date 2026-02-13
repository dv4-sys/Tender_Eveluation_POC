import json
from typing import List, Dict
from langchain_core.prompts import PromptTemplate

from .config import llm, MAX_CONTEXT_CHARS
from .normalization import normalize_keys
from .json_utils import extract_json_array

def build_context(docs):
    ctx, cites, total = [], [], 0

    for i, d in enumerate(docs, 1):
        if total >= MAX_CONTEXT_CHARS:
            break
        chunk = d.page_content[:1000]
        ctx.append(f"[S{i}] {chunk}")
        cites.append({"id": f"S{i}", **d.metadata})
        total += len(chunk)

    return "\n\n".join(ctx), cites

def evaluate_all_criteria(criteria: List[Dict], store):
    query = " ".join(c["description"] for c in criteria)
    docs = store.similarity_search(query, k=15)

    if not docs:
        return [], []

    context, citations = build_context(docs)

    prompt = PromptTemplate(
        input_variables=["criteria", "context"],
        template="""
Evaluate each criterion independently.

Return ONLY JSON array:
[
  {
    "criterion_name": "string",
    "compliant": true | false,
    "confidence": "High|Medium|Low",
    "reasoning": "string",
    "gaps": "string"
  }
]

Criteria:
{criteria}

Context:
{context}
"""
    )

    raw = llm.invoke(prompt.format(
        criteria=json.dumps(criteria),
        context=context
    ))

    return normalize_keys(extract_json_array(raw)), citations
