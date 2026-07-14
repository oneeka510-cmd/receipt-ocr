"""
receipt_parser.py

Takes the raw text description produced by the vision model
(ocr_extractor.py) and turns it into:
  1. a structured Python dict (via a local text LLM), then
  2. a pandas DataFrame, one row per line item.
"""

import json
import re

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


RECEIPT_SCHEMA_PROMPT = """Extract the following receipt details from the text below and
return them as a single, valid JSON object. Do not include any explanation,
markdown formatting, or code fences - return ONLY the raw JSON.

Fields to extract:
- company (string or null)
- bill_to (string or null)
- receipt_number (string or null)
- date (string or null)
- items (list of objects, each with: description, quantity, unit_price, total)
- subtotal (number or null)
- tax (number or null)
- total (number or null)
- payment_method (string or null)

If a field is not present in the text, use null (or an empty list for items).
Use numbers (not strings) for all monetary values.

Text to extract from:
{raw_text}
"""


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences if the model added them anyway."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def parse_receipt_to_json(raw_text: str, model: str = "llama3") -> dict:
    """
    Turn the raw text description of a receipt into a structured dict.

    Args:
        raw_text: Plain-text receipt description (output of extract_receipt_text).
        model: Name of the local Ollama text model used to do the structuring.

    Returns:
        A dict with the parsed receipt fields.

    Raises:
        ValueError: If the model's output couldn't be parsed as JSON.
    """
    prompt = ChatPromptTemplate.from_template(RECEIPT_SCHEMA_PROMPT)
    llm = ChatOllama(model=model, temperature=0)
    chain = prompt | llm | StrOutputParser()

    raw_response = chain.invoke({"raw_text": raw_text})
    cleaned = _strip_code_fences(raw_response).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Could not parse the model's output as JSON.\n"
            f"Raw model output was:\n{raw_response}"
        ) from exc


def receipt_json_to_dataframe(receipt: dict) -> pd.DataFrame:
    """
    Flatten a parsed receipt dict into a pandas DataFrame - one row per
    line item, with the receipt-level fields repeated on every row.

    Args:
        receipt: Dict produced by parse_receipt_to_json.

    Returns:
        A pandas DataFrame with one row per item.
    """
    items = receipt.get("items") or []
    items_df = (
        pd.DataFrame(items)
        if items
        else pd.DataFrame(columns=["description", "quantity", "unit_price", "total"])
    )

    header_fields = {
        "company": receipt.get("company"),
        "bill_to": receipt.get("bill_to"),
        "receipt_number": receipt.get("receipt_number"),
        "date": receipt.get("date"),
        "subtotal": receipt.get("subtotal"),
        "tax": receipt.get("tax"),
        "total": receipt.get("total"),
        "payment_method": receipt.get("payment_method"),
    }

    for key, value in header_fields.items():
        items_df[key] = value

    return items_df
