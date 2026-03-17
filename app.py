import os
import json
import pandas as pd
import requests
import streamlit as st
from bedrock_llm import BedrockLLM

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 95vw;
        padding-left: 2vw;
        padding-right: 2vw;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 85vh;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Tender Analyzer")
api_timeout_seconds = int(os.getenv("API_TIMEOUT_SECONDS", "0"))
results = None


@st.cache_resource
def get_llm():
    return BedrockLLM()


def fallback_summary(payload: dict) -> str:
    total_requirements = payload.get("total_requirements", 0)
    bidder_scores = payload.get("bidder_scores", {})
    bidder_status_counts = payload.get("bidder_status_counts", {})
    ranked = sorted(bidder_scores.items(), key=lambda x: x[1], reverse=True)

    if not ranked:
        return "No evaluation data available to summarize."

    lines = ["- Top evaluated bidder based on current scoring: **{}** (**{} / {}**).".format(
        ranked[0][0], ranked[0][1], total_requirements
    )]
    for bidder_name, score in ranked:
        counts = bidder_status_counts.get(bidder_name, {})
        lines.append(
            "- **{}**: {} Compliant, {} Partially Compliant, {} Non-Compliant, {} Not Found. Total: **{} / {}**.".format(
                bidder_name,
                counts.get("Compliant", 0),
                counts.get("Partially Compliant", 0),
                counts.get("Non-Compliant", 0),
                counts.get("Not Found", 0),
                score,
                total_requirements,
            )
        )
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def generate_llm_summary(payload_json: str) -> str:
    payload = json.loads(payload_json)
    prompt = f"""
You are a tender evaluation analyst.
Create a concise "Evaluation Summary and Insights" in markdown using only the input data.

Required structure:
1) A 2-3 sentence overall summary.
2) Bullet list for each bidder with strengths/risks grounded in status counts and score.
3) Bullet list of top risk criteria (from risk_criteria).
4) One closing recommendation line based only on score and compliance evidence trends.

Rules:
- Do not invent facts or numbers.
- Use exact scores and counts from data.
- Keep it under 220 words.
- Do not output JSON.

Data:
{json.dumps(payload, ensure_ascii=True)}
"""
    return get_llm().invoke(prompt)




left_col, right_col = st.columns([1, 2])

with left_col:
    with st.container(border=True):
        st.subheader("Upload Documents")
        tender = st.file_uploader("Upload Tender PDF", type="pdf")
        bidders = st.file_uploader("Upload Bidder PDFs", type="pdf", accept_multiple_files=True)
        analyze_clicked = st.button("Analyze")

        if analyze_clicked:
            if not tender or not bidders:
                st.error("Please upload tender and at least one bidder PDF.")
                st.stop()

            files = [("tender", (tender.name, tender, "application/pdf"))]
            for bidder in bidders:
                files.append(("bidders", (bidder.name, bidder, "application/pdf")))

            timeout = None if api_timeout_seconds <= 0 else api_timeout_seconds

            try:
                with st.spinner("Analyzing..."):
                    response = requests.post(
                        "http://127.0.0.1:8000/analyze/",
                        files=files,
                        timeout=timeout,
                    )
            except requests.exceptions.ReadTimeout:
                st.error(
                    "Analysis is taking too long and timed out. "
                    "Set API_TIMEOUT_SECONDS=0 for no timeout, or increase it."
                )
                st.stop()
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

            if response.status_code != 200:
                st.error(f"Backend Error: {response.text}")
                st.stop()

            data = response.json()
            results = data.get("results", {})

with right_col:
    with st.container(border=True):
        st.subheader("Compliance Matrix")

        if not results:
            st.info("Upload files and click Analyze to generate the compliance matrix.")
        else:
            st.caption(
                "Scoring rule: `1` for `Compliant`, `0.5` for `Partially Compliant`, `0` for `Non-Compliant` / `Not Found`."
            )

            # Flatten and index results by requirement type.
            req_index = {}
            for bidder_name, bidder_results in results.items():
                for item in bidder_results:
                    req_typ = (
                        item.get("evaluation_criterion")
                        or item.get("requirement_id")
                        or item.get("requirement_type")
                        or "UNSPECIFIED"
                    )
                    if req_typ not in req_index:
                        req_index[req_typ] = {
                            "requirement_criterion": req_typ,
                            "requirement_details": item.get("description", ""),
                            "by_bidder": {},
                        }
                    req_index[req_typ]["by_bidder"][bidder_name] = item

            def _req_sort_key(req_type: str):
                if "-" in req_type:
                    prefix, suffix = req_type.rsplit("-", 1)
                    if suffix.isdigit():
                        return (prefix, int(suffix))
                return (req_type, 0)

            ordered_req_types = sorted(req_index.keys(), key=_req_sort_key)
            bidder_names = list(results.keys())
            bidder_scores = {name: 0 for name in bidder_names}
            bidder_status_counts = {
                name: {
                    "Compliant": 0,
                    "Partially Compliant": 0,
                    "Non-Compliant": 0,
                    "Not Found": 0,
                }
                for name in bidder_names
            }
            criterion_status = {}

            table_rows = []
            for req_type in ordered_req_types:
                entry = req_index[req_type]
                row = {
                    "Evaluation Criterion": req_type,
                    "Requirement Details": entry["requirement_details"],
                }
                criterion_status[req_type] = {}

                for bidder_name in bidder_names:
                    item = entry["by_bidder"].get(bidder_name, {})
                    status = item.get("status", "Not Found")
                    score = 1 if status == "Compliant" else 0.5 if status == "Partially Compliant" else 0
                    bidder_scores[bidder_name] += score
                    if status not in bidder_status_counts[bidder_name]:
                        status = "Not Found"
                    bidder_status_counts[bidder_name][status] += 1
                    criterion_status[req_type][bidder_name] = status

                    evidence_list = item.get("evidence_found", [])
                    first_evidence = ""
                    if evidence_list:
                        first_evidence = str(evidence_list[0].get("text", "")).strip()
                    gap = str(item.get("gap_analysis", "")).strip()

                    note = first_evidence or gap or "No supporting note"
                    note = " ".join(note.split())[:120]
                    row[bidder_name] = f"{score} ({status}: {note})"

                table_rows.append(row)

            total_row = {
                "Evaluation Criterion": "TOTAL SCORE",
                "Requirement Details": "",
            }
            for bidder_name in bidder_names:
                total_row[bidder_name] = f"{bidder_scores[bidder_name]} / {len(ordered_req_types)}"
            table_rows.append(total_row)

            df = pd.DataFrame(table_rows)
            table_html = df.to_html(index=False, escape=True)
            st.markdown(
                """
                <style>
                .compliance-table-wrap {
                    max-height: 1000px;
                    overflow: auto;
                    border: 1px solid #d9d9d9;
                    border-radius: 8px;
                }
                .compliance-table-wrap table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9rem;
                }
                .compliance-table-wrap thead th {
                    position: sticky;
                    top: 0;
                    background: #f3f3f3;
                    z-index: 1;
                }
                .compliance-table-wrap th,
                .compliance-table-wrap td {
                    border: 1px solid #d9d9d9;
                    padding: 8px;
                    text-align: left;
                    vertical-align: top;
                }
                .compliance-table-wrap tbody td:first-child {
                    font-weight: 700;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='compliance-table-wrap'>{table_html}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### Evaluation Summary and Insights")
            total_requirements = len(ordered_req_types)
            if total_requirements > 0:
                risk_criteria = []
                for req_type in ordered_req_types:
                    statuses = criterion_status.get(req_type, {}).values()
                    risk_count = sum(1 for s in statuses if s != "Compliant")
                    if risk_count > 0:
                        risk_criteria.append({
                            "requirement_criterion": req_type,
                            "risk_count": risk_count,
                            "total_bidders": len(bidder_names),
                            "requirement_details": req_index[req_type]["requirement_details"][:180],
                        })
                risk_criteria = sorted(risk_criteria, key=lambda x: x["risk_count"], reverse=True)[:5]

                summary_payload = {
                    "total_requirements": total_requirements,
                    "scoring_rule": {
                        "Compliant": 1,
                        "Partially Compliant": 0.5,
                        "Non-Compliant": 0,
                        "Not Found": 0,
                    },
                    "bidder_scores": bidder_scores,
                    "bidder_status_counts": bidder_status_counts,
                    "risk_criteria": risk_criteria,
                }

                summary_text = ""
                try:
                    summary_text = generate_llm_summary(json.dumps(summary_payload, sort_keys=True))
                except Exception:
                    summary_text = fallback_summary(summary_payload)

                st.markdown(summary_text)
