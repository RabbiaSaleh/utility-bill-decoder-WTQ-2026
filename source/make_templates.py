"""Create empty level1.csv / level2.csv templates for local testing."""

import csv
from pathlib import Path

# Example bill IDs matching the guide
BILL_IDS = [
    "IESCO_0002", "KESC_0002", "KESC_0003", "KESC_0005", "KESC_0007",
    "LESCO_0001", "LESCO_0003", "LESCO_0004", "LESCO_0007", "LESCO_0008",
]

# Placeholder questions (real ones come in the test set at 12:30)
SAMPLE_QUESTIONS = [
    ("Q1", "How much of my bill is taxes?"),
    ("Q2", "How many units did I consume this month?"),
    ("Q3", "What is the due date for payment without surcharge?"),
    ("Q4", "What is my previous dues / arrears amount?"),
    ("Q5", "How much is the fuel price adjustment?"),
    ("Q6", "What is the total of electricity charges before taxes?"),
    ("Q7", "What is the amount payable after the due date?"),
    ("Q8", "What is my sanctioned load in kW?"),
    ("Q9", "How many months in my history went above 200 units?"),
    ("Q10", "What percentage of the current bill is taxes?"),
    ("Q11", "Is there any subsidy or credit on this bill?"),
    ("Q12", "What is the difference between payable within and after due date?"),
]


def main():
    out = Path("../test/csv")
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "level1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bill_id", "json"])
        w.writeheader()
        for bid in BILL_IDS:
            w.writerow({"bill_id": bid, "json": ""})

    with open(out / "level2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["bill_id", "question_id", "question", "answer"]
        )
        w.writeheader()
        for bid in BILL_IDS:
            for qid, qtext in SAMPLE_QUESTIONS:
                w.writerow(
                    {
                        "bill_id": bid,
                        "question_id": qid,
                        "question": qtext,
                        "answer": "",
                    }
                )

    print(f"Templates written to {out}")


if __name__ == "__main__":
    main()