# 🧾 Local Receipt OCR Extractor
<video controls src="receipt_ocr_video.mp4" title="Title"></video>
Upload a photo of a receipt and get back a structured table of its
contents (company, date, line items, totals) — entirely offline, using
local LLMs served by [Ollama](https://ollama.com/). No API keys, no
cloud calls, no data leaving your machine.

## How it works

```
receipt.jpg
    │
    ▼
[1] gemma3:4b (vision model)   (ocr_extractor.py)
    reads the image, returns a plain-text description
    │
    ▼
[2] llama3 (text model)        (receipt_parser.py)
    turns that free text into a strict JSON schema
    │
    ▼
pandas DataFrame → table in the UI + downloadable CSV
```

## Project structure

```
receipt-ocr-extractor/
├── app.py              # Streamlit UI - ties everything together
├── ocr_extractor.py     # Step 1: image -> raw text (vision model)
├── receipt_parser.py    # Step 2: raw text -> JSON -> DataFrame
├── requirements.txt
├── sample_receipts/      # drop test images here
└── README.md
```

## Setup

**1. Install [Ollama](https://ollama.com/download)** and make sure it's running.

**2. Pull the two models used by this project:**
```bash
ollama pull gemma3:4b
ollama pull llama3
```

**3. Clone this repo and install Python dependencies:**
```bash
git clone https://github.com/<your-username>/receipt-ocr-extractor.git
cd receipt-ocr-extractor
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**4. Run the app:**
```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`),
upload a receipt image, and click **Extract receipt data**.

## Notes / limitations

- Accuracy depends entirely on the vision model's read of the image —
  blurry or low-res photos will produce worse results.
- Expect each extraction to take anywhere from a few seconds to over a
  minute depending on your hardware (CPU vs GPU).
- Built for **receipts**, not general documents — the JSON schema in
  `receipt_parser.py` is intentionally specific to keep extraction accurate.
- Uses `gemma3:4b` for vision rather than `llama3.2-vision`, since Ollama's
  newer engine versions currently don't support the `mllama` architecture
  that `llama3.2-vision` relies on.

## License

MIT
