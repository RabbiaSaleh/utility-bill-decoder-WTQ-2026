"""
Utility Bill Decoder – WTQ Build Track 2026

End-to-end pipeline:
  OCR → structured Level-1 extraction → Level-2 Q&A → CSV generation

Usage examples:
  # Process training bills (exploration)
  python main.py --bills-dir ../train/bills --mode train

  # Produce competition CSVs from test set
  python main.py --bills-dir ../test/bills \\
                 --level1-template ../test/csv/level1.csv \\
                 --level2-template ../test/csv/level2.csv \\
                 --output-dir ../output

  # Single bill debug
  python main.py --single ../train/bills/KESC_0008.png
"""

from __future__ import annotations
import time
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from copy import deepcopy

from schema import EMPTY_TEMPLATE, to_json_string
from dotenv import load_dotenv

load_dotenv()

from extractor import extract_bill
from llm_client import LLMClient
from ocr_engine import extract_text
from qa_engine import answer_question
from schema import to_json_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def process_single(image_path: Path, llm: LLMClient) -> dict:
    data = extract_bill(image_path, llm=llm)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_level1(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bill_id", "json"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Wrote %s (%d rows)", out_path, len(rows))


def write_level2(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["bill_id", "question_id", "question", "answer"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Wrote %s (%d rows)", out_path, len(rows))


def run_test_pipeline(
    bills_dir: Path,
    level1_template: Path,
    level2_template: Path,
    output_dir: Path,
    llm: LLMClient,
) -> None:
    """
    Generate completed level1.csv and level2.csv from templates.

    Missing bill images do not stop the pipeline. Instead, a complete
    null Level-1 schema is written for that bill.
    """

    l1_rows = load_csv(level1_template)
    l2_rows = load_csv(level2_template)

    bill_cache: dict[str, dict] = {}
    ocr_cache: dict[str, str] = {}
    missing_bill_ids: set[str] = set()

    allowed_extensions = {".png", ".jpg", ".jpeg"}

    # ---------------------------------------------------------
    # Level 1
    # ---------------------------------------------------------
    completed_l1 = []

    for row in l1_rows:
        bill_id = row["bill_id"].strip()

        candidates = [
            path
            for path in bills_dir.glob(f"{bill_id}.*")
            if path.suffix.lower() in allowed_extensions
        ]

        # Missing image: write full null schema and continue
        if not candidates:
            logger.warning(
                "Bill image not found for %s. "
                "Writing a complete null schema.",
                bill_id,
            )

            data = empty_bill_record()

            bill_cache[bill_id] = data
            ocr_cache[bill_id] = ""
            missing_bill_ids.add(bill_id)

            completed_l1.append(
                {
                    "bill_id": bill_id,
                    "json": to_json_string(data),
                }
            )

            continue

        image_path = candidates[0]

        try:
            data = extract_bill(
                image_path,
                llm=llm,
            )

            # Safety fallback in case extractor returns None or {}
            if not isinstance(data, dict) or not data:
                logger.warning(
                    "Empty extraction received for %s. "
                    "Using complete null schema.",
                    bill_id,
                )
                data = empty_bill_record()

            bill_cache[bill_id] = data

            # OCR is used again here for Level-2 context.
            # This is safe, although it can later be optimized to avoid
            # duplicate OCR execution.
            ocr_result = extract_text(image_path)
            ocr_cache[bill_id] = ocr_result.get("raw_text", "")

            completed_l1.append(
                {
                    "bill_id": bill_id,
                    "json": to_json_string(data),
                }
            )

            logger.info("Level-1 done: %s", bill_id)

        except Exception as exc:
            logger.exception(
                "Extraction failed for %s: %s. "
                "Writing complete null schema.",
                bill_id,
                exc,
            )

            data = empty_bill_record()

            bill_cache[bill_id] = data
            ocr_cache[bill_id] = ""
            missing_bill_ids.add(bill_id)

            completed_l1.append(
                {
                    "bill_id": bill_id,
                    "json": to_json_string(data),
                }
            )

    write_level1(
        completed_l1,
        output_dir / "level1.csv",
    )

    # ---------------------------------------------------------
    # Level 2
    # ---------------------------------------------------------
    completed_l2 = []

    for row in l2_rows:
        bill_id = row["bill_id"].strip()
        question_id = row["question_id"].strip()
        question = row["question"]

        data = bill_cache.get(bill_id)

        if data is None:
            logger.warning(
                "No cached data for %s. "
                "Using complete null schema.",
                bill_id,
            )

            data = empty_bill_record()
            bill_cache[bill_id] = data
            ocr_cache[bill_id] = ""
            missing_bill_ids.add(bill_id)

        # Do not call the LLM with a missing image.
        # Return a truthful non-empty answer instead.
        if bill_id in missing_bill_ids:
            answer = (
                "The bill image was not available, so this information "
                "cannot be determined from the bill."
            )
        else:
            ocr_text = ocr_cache.get(bill_id, "")

            try:
                answer = answer_question(
                    question=question,
                    bill_data=data,
                    ocr_text=ocr_text,
                    llm=llm,
                )

                # Ensure answer cell is never empty
                if not answer or not answer.strip():
                    answer = (
                        "The bill does not provide enough readable "
                        "information to answer this question."
                    )

            except Exception as exc:
                logger.exception(
                    "Level-2 failed for %s/%s: %s",
                    bill_id,
                    question_id,
                    exc,
                )

                answer = (
                    "The bill does not provide enough readable "
                    "information to answer this question."
                )
            time.sleep(4)
        completed_l2.append(
            {
                "bill_id": bill_id,
                "question_id": question_id,
                "question": question,
                "answer": answer.strip(),
            }
        )

        logger.info(
            "Level-2 %s %s done",
            bill_id,
            question_id,
        )

    write_level2(
        completed_l2,
        output_dir / "level2.csv",
    )

    logger.info(
        "Pipeline complete. CSVs are in %s",
        output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Utility Bill Decoder")
    parser.add_argument(
        "--bills-dir", type=Path, default=Path("../test/bills"),
        help="Directory containing bill images",
    )
    parser.add_argument(
        "--level1-template", type=Path, default=Path("../test/csv/level1.csv"),
    )
    parser.add_argument(
        "--level2-template", type=Path, default=Path("../test/csv/level2.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../output"),
    )
    parser.add_argument(
        "--single", type=Path, default=None,
        help="Process a single bill image and print JSON",
    )
    parser.add_argument(
        "--mode", choices=["test", "train"], default="test",
    )
    args = parser.parse_args()

    llm = LLMClient()
    logger.info("Using %s / %s", llm.provider, llm.model)

    if args.single:
        process_single(args.single, llm)
        return

    if args.mode == "train":
        # Just extract all training bills for inspection
        bills = sorted(args.bills_dir.glob("*.*"))
        for b in bills:
            if b.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            data = extract_bill(b, llm=llm)
            out = args.output_dir / f"{b.stem}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            logger.info("Saved %s", out)
        return

    # Full test pipeline
    run_test_pipeline(
        bills_dir=args.bills_dir,
        level1_template=args.level1_template,
        level2_template=args.level2_template,
        output_dir=args.output_dir,
        llm=llm,
    )

def empty_bill_record() -> dict:
    """
    Return a fresh complete null Level-1 schema.

    deepcopy is important because EMPTY_TEMPLATE contains mutable
    list values such as charges and taxes.
    """
    return deepcopy(EMPTY_TEMPLATE)

if __name__ == "__main__":
    main()