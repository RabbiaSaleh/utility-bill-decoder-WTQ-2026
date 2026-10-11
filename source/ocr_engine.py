"""
OCR layer – extracts raw text from preprocessed bill images.
Uses Tesseract (eng). Can be extended with PaddleOCR when available.
"""

from __future__ import annotations

import pytesseract
from pytesseract import Output
import logging
from pathlib import Path
from preprocess import preprocess_for_ocr

logger = logging.getLogger(__name__)


def run_tesseract(image, lang: str = "eng", config: str = "--oem 3 --psm 6") -> str:
    """Run Tesseract and return plain text."""
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        return text
    except Exception as e:
        logger.warning("Tesseract failed: %s", e)
        return ""


def run_tesseract_detailed(
    image,
    lang: str = "eng",
) -> list[dict]:
    """Return word-level boxes with confidence scores."""
    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=Output.DICT,
        )

        words = []
        n = len(data["text"])

        for i in range(n):
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                continue

            txt = data["text"][i].strip()

            if conf > 30 and txt:
                words.append(
                    {
                        "text": txt,
                        "conf": conf,
                        "left": int(data["left"][i]),
                        "top": int(data["top"][i]),
                        "width": int(data["width"][i]),
                        "height": int(data["height"][i]),
                    }
                )

        return words

    except Exception as e:
        logger.warning("Tesseract detailed failed: %s", e)
        return []

def extract_text(image_path: str | Path) -> dict:
    """
    Main entry: preprocess + OCR.
    Returns dict with raw_text, binary_text, and word boxes.
    """
    image_path = str(image_path)
    color, binary = preprocess_for_ocr(image_path)

    # Multiple PSM modes improve recall on varied layouts
    texts = []
    for psm in (6, 4, 3):
        cfg = f"--oem 3 --psm {psm}"
        t = run_tesseract(color, config=cfg)
        if t.strip():
            texts.append(t)

    # Also try on binary
    t_bin = run_tesseract(binary, config="--oem 3 --psm 6")
    if t_bin.strip():
        texts.append(t_bin)

    # Prefer the longest non-empty extraction
    raw_text = max(texts, key=len) if texts else ""

    words = run_tesseract_detailed(color)

    return {
        "raw_text": raw_text,
        "words": words,
        "image_path": image_path,
    }