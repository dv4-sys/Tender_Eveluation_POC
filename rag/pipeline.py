from typing import Dict
from .logging import log
from .pdf_loader import extract_text_from_pdf, extract_many_pdfs
from .chunking import chunk_documents
from .criteria import extract_tender_criteria
from .evaluation import evaluate_all_criteria
from .vectorstore import load_or_create_store, hash_docs

def validate_bidder(name, criteria, store):
    results, citations = evaluate_all_criteria(criteria, store)
    by_name = {r["criterion_name"]: r for r in results}

    total, failed, evaluations = 0, False, []

    for c in criteria:
        r = by_name.get(c["criterion_name"], {})
        compliant = bool(r.get("compliant", False))

        if c["mandatory"] and not compliant:
            failed = True

        score = 10 if compliant else 3
        total += score

        evaluations.append({
            "criterion": c["criterion_name"],
            "mandatory": c["mandatory"],
            "compliant": compliant,
            "confidence": r.get("confidence", "Low"),
            "reasoning": r.get("reasoning", ""),
            "gaps": r.get("gaps", ""),
            "score": score,
            "citations": citations
        })

    return {
        "bidder": name,
        "score": min(100, int(total / (len(criteria) * 10) * 100)),
        "mandatory_failed": failed,
        "evaluations": evaluations
    }

def run_poc(tender_pdf: str, bidder_pdfs: Dict[str, str], progress_cb=None):
    log("Reading tender", progress_cb)
    tender_docs = extract_text_from_pdf(tender_pdf)

    log("Extracting criteria", progress_cb)
    criteria = extract_tender_criteria(
        "\n".join(d.page_content for d in tender_docs)
    )

    log("Processing bidders", progress_cb)
    bidder_docs = extract_many_pdfs(bidder_pdfs)

    results = []

    for bidder, docs in bidder_docs.items():
        for d in docs:
            d.metadata["source"] = bidder

        chunks = chunk_documents(docs)
        store = load_or_create_store(chunks, hash_docs(chunks))

        log(f"Evaluating {bidder}", progress_cb)
        results.append(validate_bidder(bidder, criteria, store))

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    return {
        "winner": ranked[0]["bidder"],
        "ranked_bidders": ranked
    }
