Utility Bill Decoder – WTQ Build Track 2026

AI-powered pipeline that reads Pakistani electricity bills (K-Electric, LESCO, IESCO), extracts structured details, and answers customer questions in plain English.

Architecture (3 layers)

┌─────────────────┐
│  1. OCR Engine  │  OpenCV preprocess → Tesseract text + word boxes
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. Extraction   │  Multimodal LLM (image + OCR) → strict JSON schema
│    + Verify     │  Type cleaning, null handling, sum consistency checks
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. QA Engine    │  Extracted JSON + OCR + question → grounded English answer
└────────┬────────┘
         ▼
   level1.csv + level2.csv

Why this design?





Pure LLM-on-image alone often confuses meter readings, units, arrears and totals.



OCR gives a reliable text grounding; the LLM then maps fields with the schema.



A verification pass logs inconsistencies without inventing values (competition rule: extract as printed).

Requirements





Python 3.11+



One of the allowed models (see .env.example)

pip install -r requirements.txt

System dependency: Tesseract OCR

# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

Configuration

Copy .env.example → .env and set your key:

cp .env.example .env
# edit .env

Recommended free option:

MODEL_PROVIDER=google
MODEL_NAME=gemini-2.0-flash-lite   # or gemini-2.0-flash
API_KEY=your-google-ai-studio-key

Allowed models only (competition rule):





Claude Haiku 4.5 (claude-haiku-4-5)



Gemini 3.5 / 3.1 Flash-Lite 



GPT-4.1 mini / GPT-5 mini

How to run

1. Explore training bills

python main.py --bills-dir ../train/bills --mode train --output-dir ../output

2. Generate competition CSVs (test set)

Place the 10 test bills in ../test/bills/ and the two templates in ../test/csv/.

python main.py \
  --bills-dir ../test/bills \
  --level1-template ../test/csv/level1.csv \
  --level2-template ../test/csv/level2.csv \
  --output-dir ../output

This produces:





../output/level1.csv – 10 rows, each with a Level-1 JSON object



../output/level2.csv – 120 rows, each with a grounded English answer

3. Debug a single bill

python main.py --single ../train/bills/KESC_0008.png

Project layout

source/
├── main.py           # CLI entry point
├── preprocess.py     # Deskew, CLAHE, denoise, binarize
├── ocr_engine.py     # Tesseract multi-PSM extraction
├── schema.py         # JSON schema, cleaning, validation
├── llm_client.py     # Unified Google / OpenAI / Anthropic client
├── extractor.py      # Level-1 pipeline
├── qa_engine.py      # Level-2 answers
├── requirements.txt
├── .env.example
└── README.md

AI usage







Category



Used?



Details





Allowed competition LLM



Yes



Configured via MODEL_PROVIDER / MODEL_NAME





Vision / multimodal



Yes



Bill image sent together with OCR text





Coding assistants



Yes



Grok (xAI) used to design & write this code





Local OCR



Yes



Tesseract 5 + OpenCV preprocessing





Prohibited services



No



No Azure Invoice, Textract, Mindee, etc.

Notes for judges





All numeric values are taken as printed (no recalculation or rounding).



Credits / subsidies are negative.



payable_after_due_date selects the highest late-payment amount when several appear.



Missing / grey-boxed fields become null.



Personal identifiers are never returned.



Answers are grounded exclusively in the bill; estimates are labelled.

Reproducibility

python --version          # 3.11+
pip install -r requirements.txt
# set API_KEY in .env
python main.py --bills-dir <path-to-test-bills> ...