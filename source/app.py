"""
Quick Streamlit demo UI for Utility Bill Decoder
Run: streamlit run source/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Utility Bill Decoder | WTQ 2026",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Utility Bill Decoder")
st.caption("WTQ Build Track 2026 — OCR → Extraction → Q&A Pipeline")

with st.sidebar:
    st.header("Settings")
    bills_dir = st.text_input("Bills folder", value="train/bills")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("1. OCR (Tesseract)")
    st.markdown("2. Structured Extraction (LLM)")
    st.markdown("3. Question Answering")

bills_path = Path(bills_dir)
available = []
if bills_path.exists():
    available = sorted(
        [p for p in bills_path.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

if not available:
    st.warning(f"No bill images found in `{bills_dir}`. Change the path in the sidebar.")
    st.stop()

bill_names = [p.name for p in available]
selected_name = st.selectbox("Select a bill", bill_names)
selected_path = bills_path / selected_name

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Bill Image")
    st.image(str(selected_path), use_container_width=True)

with col2:
    st.subheader("Level 1 — Extracted JSON")
    if st.button("Extract Bill Details", type="primary"):
        with st.spinner("Running OCR + LLM extraction..."):
            try:
                from extractor import extract_bill
                from llm_client import LLMClient

                llm = LLMClient()
                data = extract_bill(selected_path, llm=llm)
                st.session_state["last_data"] = data
                st.session_state["last_bill"] = selected_name
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()

    if "last_data" in st.session_state and st.session_state.get("last_bill") == selected_name:
        data = st.session_state["last_data"]
        st.json(data)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Units", data.get("units_consumed"))
        m2.metric("Current Bill", data.get("current_bill"))
        m3.metric("Taxes", data.get("total_taxes"))
        m4.metric("After Due Date", data.get("payable_after_due_date"))

st.markdown("---")
st.subheader("Level 2 — Ask a Question")

sample_questions = [
    "How much of my bill is taxes?",
    "How many units did I consume this month?",
    "What is the due date for payment without surcharge?",
    "What is my previous dues / arrears amount?",
    "What is the amount payable after the due date?",
    "How many months in my history went above 200 units?",
]

q = st.selectbox("Sample question", sample_questions)
custom_q = st.text_input("Or type your own question")
final_q = custom_q.strip() if custom_q.strip() else q

if st.button("Get Answer"):
    if "last_data" not in st.session_state:
        st.warning("Extract the bill first (Level 1).")
    else:
        with st.spinner("Generating answer..."):
            try:
                from qa_engine import answer_question
                from llm_client import LLMClient
                from ocr_engine import extract_text

                llm = LLMClient()
                ocr = extract_text(selected_path)
                ans = answer_question(
                    final_q,
                    st.session_state["last_data"],
                    ocr_text=ocr.get("raw_text", ""),
                    llm=llm,
                )
                st.success(ans)
            except Exception as e:
                st.error(f"QA failed: {e}")

st.markdown("---")
st.caption("Pipeline: OpenCV preprocess → Tesseract OCR → Multimodal LLM extraction → Grounded Q&A")
