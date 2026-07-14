"""
ocr_extractor.py

Uses a local vision-capable LLM (via Ollama) to "read" a receipt image
and return a plain-text description of everything on it. This is the
OCR step: image in, raw text out.
"""

import ollama


def extract_receipt_text(image_path: str, model: str = "gemma3:4b") -> str:
    """
    Send a receipt image to a local Ollama vision model and get back
    a plain-text description of everything on the receipt.

    Args:
        image_path: Path to the receipt image on disk (jpg/jpeg/png).
        model: Name of the Ollama vision model to use.

    Returns:
        The model's raw text description of the receipt contents.

    Raises:
        RuntimeError: If Ollama can't be reached or the model isn't
            available locally (e.g. Ollama isn't running, or the model
            hasn't been pulled yet).
    """
    prompt = (
        "You are looking at a receipt or invoice. Extract ALL text and "
        "data from the image as accurately as possible, including the "
        "store/company name, date, every line item with its price, "
        "subtotal, tax, and total. List everything you can read, even "
        "if you're not sure what it means."
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not get a response from Ollama model '{model}'. "
            f"Make sure Ollama is running and the model is pulled "
            f"(try: ollama pull {model})."
        ) from exc

    return response["message"]["content"].strip()
