"""
app.py

Streamlit front-end for the receipt OCR extractor.

Flow:
    upload image -> ocr_extractor (vision LLM reads it) ->
    receipt_parser (text LLM structures it into JSON) -> table + CSV

Run with:
    streamlit run app.py

Requires Ollama running locally with these models pulled:
    ollama pull gemma3:4b
    ollama pull llama3
"""

import tempfile
from pathlib import Path

import streamlit as st

from ocr_extractor import extract_receipt_text
from receipt_parser import parse_receipt_to_json, receipt_json_to_dataframe


st.set_page_config(page_title="Receipt OCR Extractor", page_icon="🧾", layout="wide")

st.title("🧾 Local Receipt OCR Extractor")
st.caption(
    "Upload a photo of a receipt. Everything runs locally through Ollama - "
    "no data leaves your machine."
)

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    vision_model = st.text_input("Vision model", value="gemma3:4b")
with col2:
    text_model = st.text_input("Text model (for structuring JSON)", value="llama3")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded receipt", width=350)

    if st.button("Extract receipt data", type="primary"):
        # Ollama needs a file path, so save the upload to a temp file first
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner(f"Reading the image with {vision_model}..."):
                raw_text = extract_receipt_text(tmp_path, model=vision_model)

            with st.expander("Raw model output (from vision model)"):
                st.text(raw_text)

            with st.spinner(f"Structuring the data with {text_model}..."):
                receipt_data = parse_receipt_to_json(raw_text, model=text_model)

            st.subheader("Structured JSON")
            st.json(receipt_data)

            st.subheader("Line items")
            df = receipt_json_to_dataframe(receipt_data)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download as CSV",
                data=csv,
                file_name=f"{Path(uploaded_file.name).stem}_receipt.csv",
                mime="text/csv",
            )

        except RuntimeError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(f"Couldn't structure the receipt data: {exc}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
else:
    st.info("Upload a receipt image above to get started.")
