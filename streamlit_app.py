import streamlit as st
import tempfile
from pathlib import Path
from rag_engine import run_poc

st.set_page_config(layout="wide")
st.title("📑 Tender Evaluation")

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def save_uploaded_files(uploaded_files, tmp_dir: Path):
    """Save uploaded files and return {stem: filepath}"""
    file_paths = {}
    for file in uploaded_files:
        file_path = tmp_dir / file.name
        file_path.write_bytes(file.read())
        file_paths[file_path.stem] = str(file_path)
    return file_paths

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

tender = st.file_uploader(
    "Tender PDF",
    type="pdf",
    help="Upload the tender document",
)

bidders = st.file_uploader(
    "Bidder PDFs",
    type="pdf",
    accept_multiple_files=True,
    help="Upload bidder proposals",
)

# ------------------------------------------------------------------
# ACTION
# ------------------------------------------------------------------

if st.button("Run Evaluation", type="primary"):

    if not tender:
        st.error("⚠️ Please upload a tender document")
        st.stop()

    if not bidders:
        st.error("⚠️ Please upload at least one bidder document")
        st.stop()

    with st.spinner("Evaluating tenders..."):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)

                # Save tender
                tender_path = tmp_path / tender.name
                tender_path.write_bytes(tender.read())

                # Save bidders
                bidder_paths = save_uploaded_files(bidders, tmp_path)

                # Run RAG pipeline
                result = run_poc(str(tender_path), bidder_paths)

        except ValueError as e:
            st.error(f"❌ Evaluation failed: {e}")
            st.stop()

        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------

    if not result.get("ranked_bidders"):
        st.warning("No valid bidder evaluations returned.")
        st.stop()

    winner = result.get("winner") or "No winner (mandatory criteria failed)"
    st.success(f"🏆 Winner: **{winner}**")
    st.divider()

    for i, bidder_result in enumerate(result["ranked_bidders"], start=1):
        badge = (
            "🥇" if i == 1 else
            "🥈" if i == 2 else
            "🥉" if i == 3 else
            "📋"
        )

        with st.expander(
            f"{badge} {bidder_result['bidder']} - Score: {bidder_result['score']}/100"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Score", f"{bidder_result['score']}/100")

            with col2:
                status = (
                    "🔴 Failed"
                    if bidder_result.get("mandatory_failed")
                    else "🟢 Passed"
                )
                st.metric("Mandatory Requirements", status)

            st.subheader("Evaluations")

            for eval_item in bidder_result.get("evaluations", []):
                compliance = "✅" if eval_item.get("compliant") else "❌"
                mandatory_tag = (
                    "[MANDATORY]" if eval_item.get("mandatory") else "[OPTIONAL]"
                )

                st.markdown(
                    f"### {compliance} {eval_item['criterion']} {mandatory_tag}"
                )

                st.write(
                    f"**Reasoning:** {eval_item.get('reasoning', 'N/A')}"
                )

                if eval_item.get("gaps"):
                    st.warning(f"**Gaps:** {eval_item['gaps']}")
