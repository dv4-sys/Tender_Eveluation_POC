from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from .config import llm
from .json_utils import extract_json_array
from .normalization import normalize_keys

def extract_tender_criteria(tender_text: str) -> List[Dict]:
    tender_text = tender_text[:15000]

    prompt = PromptTemplate(
        input_variables=["tender"],
        template="""
Extract evaluation criteria.

Return ONLY valid JSON array:
[
  {
    "criterion_name": "string",
    "description": "string",
    "mandatory": true | false
  }
]

Tender:
{tender}
"""
    )

    raw = llm.invoke(prompt.format(tender=tender_text))
    criteria = normalize_keys(extract_json_array(raw))

    if not criteria:
        raise ValueError("No criteria extracted")

    return criteria
