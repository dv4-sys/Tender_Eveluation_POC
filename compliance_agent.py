# compliance_agent.py
from bedrock_llm import BedrockLLM

llm = BedrockLLM()

def evaluate_requirement(requirement, retrieved_docs, bidder_name):

    context = "\n\n".join(
        [
            f"(Page {doc.metadata.get('page', 'N/A')})\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    requirement_type = str(requirement.get("requirement_type", "")).strip() if isinstance(requirement, dict) else ""
    requirement_description = str(requirement.get("description", "")).strip() if isinstance(requirement, dict) else str(requirement)
    requirement_acceptance_criteria = str(requirement.get("acceptance_criteria", "")).strip() if isinstance(requirement, dict) else ""

    prompt = f"""
You are a professional Tender compliance auditor.

Bidder should fulfil the Technical and Financial eligibility requirements.

Your task is to evaluate if the bidder meets the following requirement based on the provided evidence from bidder documents.

Analyze the requirement against bidder documents.

Requirement:
Type: {requirement_type}
Description: {requirement_description}
Acceptance Criteria: {requirement_acceptance_criteria}

Bidder:
{bidder_name}

Bidder Document Evidence:
{context}

Instructions:
- Compare requirement carefully with evidence.
- If fully satisfied → Compliant
- If partially satisfied → Partially Compliant
- If not satisfied → Non-Compliant
- If no evidence found → Not Found
- Quote exact supporting text from documents.
- Mention page numbers.
- Explain gaps clearly if any.

Return ONLY valid JSON:

{{
  "evaluation_criterion": "SAP Implementation Experience (Technical) | Average Annual Turnover (Financial) | ...",
  "requirement_details": "{requirement_description}",
  "acceptance_criteria": "{requirement_acceptance_criteria}",
  "status": "Compliant | Partially Compliant | Non-Compliant | Not Found",
  "score": 1 | 0.5 | 0,
  "matrix_note": "One-line concise reason for matrix cell",
  "detailed_analysis": "Step-by-step explanation of evaluation",
  "evidence_found": [
      {{
        "page": 0,
        "text": "exact quoted snippet from bidder PDF"
      }}
  ],
  "gap_analysis": "Explain missing elements if any"
}}

Scoring rule:
- If status is Compliant, score must be 1.
- For Partially Compliant, score must be 0.5.
- For Non-Compliant / Not Found, score must be 0.
"""
    raw_result = llm.invoke_json(prompt)

    # Normalize output to keep backend/UI stable even when LLM response is imperfect.
    status = str(raw_result.get("status", "Not Found")).strip()
    if status not in {"Compliant", "Partially Compliant", "Non-Compliant", "Not Found"}:
        status = "Not Found"

    score = 1 if status == "Compliant" else 0.5 if status == "Partially Compliant" else 0
    matrix_note = str(raw_result.get("matrix_note", "")).strip()
    if not matrix_note:
        first_evidence = ""
        evidence = raw_result.get("evidence_found", [])
        if isinstance(evidence, list) and evidence:
            first_evidence = str(evidence[0].get("text", "")).strip()
        gap = str(raw_result.get("gap_analysis", "")).strip()
        matrix_note = first_evidence or gap or "No supporting note"

    result = {
        "evaluation_criterion": raw_result.get("evaluation_criterion") or requirement_type,
        "requirement_details": raw_result.get("requirement_details") or requirement_description,
        "acceptance_criteria": raw_result.get("acceptance_criteria") or requirement_acceptance_criteria,
        "status": status,
        "score": score,
        "matrix_note": " ".join(matrix_note.split())[:180],
        "detailed_analysis": raw_result.get("detailed_analysis", ""),
        "evidence_found": raw_result.get("evidence_found", []),
        "gap_analysis": raw_result.get("gap_analysis", ""),
    }

    print("Evaluation Result:", result)
    return result
