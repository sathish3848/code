"""
Evaluation Script for Annual Report Q&A Assistant.
Runs 10 financial test questions against an indexed report and checks
retrieval accuracy, page citations and hallucination suppression.
Prints a Markdown table you can paste into the README.

Usage:
    uv run python evaluate.py                    # search all indexed reports
    uv run python evaluate.py "report_name.pdf"  # restrict to one report
"""

import sys
import time
from typing import List, Dict, Any, Optional

from rag_chain import FinancialRAGChain

# NOTE: expected answers and pages below were written for the TCS FY2024 report.
# If you evaluate a different PDF, update expected_answer / expected_pages.
TEST_SUITE = [
    {
        "id": 1,
        "question": "What was the total revenue in FY2024?",
        "expected_answer": "₹240,893 crore ($29.1 billion), growth of 6.8% YoY.",
        "expected_pages": [1],
        "category": "Key Financials",
    },
    {
        "id": 2,
        "question": "What are the main risks mentioned by management?",
        "expected_answer": "Macroeconomic uncertainty, GenAI disruption, cybersecurity, geopolitical conflicts, forex volatility, talent retention.",
        "expected_pages": [3],
        "category": "Risk Management",
    },
    {
        "id": 3,
        "question": "How many employees does the company have?",
        "expected_answer": "601,546 employees as of March 31, 2024.",
        "expected_pages": [4],
        "category": "Human Capital",
    },
    {
        "id": 4,
        "question": "What is the dividend per share?",
        "expected_answer": "Total dividend of ₹73.00 per share (interim ₹27 + special ₹18 + final ₹28).",
        "expected_pages": [5],
        "category": "Shareholder Returns",
    },
    {
        "id": 5,
        "question": "What was the operating profit (EBIT) and operating margin in FY2024?",
        "expected_answer": "Operating profit ₹59,314 crore, operating margin 24.6%.",
        "expected_pages": [1],
        "category": "Key Financials",
    },
    {
        "id": 6,
        "question": "What was the net profit (profit after tax) for FY2024?",
        "expected_answer": "Net profit (PAT) ₹46,099 crore (up 9.0% YoY).",
        "expected_pages": [1],
        "category": "Key Financials",
    },
    {
        "id": 7,
        "question": "What percentage of the workforce are women professionals?",
        "expected_answer": "35.6% of the workforce (over 214,000 women).",
        "expected_pages": [4],
        "category": "Human Capital",
    },
    {
        "id": 8,
        "question": "What was the total value of the share buyback in FY2024?",
        "expected_answer": "₹17,000 crore completed in December 2023 at ₹4,150 per share.",
        "expected_pages": [5],
        "category": "Shareholder Returns",
    },
    {
        "id": 9,
        "question": "What was the company's long-term debt as of March 31, 2024?",
        "expected_answer": "Zero long-term debt (unencumbered balance sheet).",
        "expected_pages": [6],
        "category": "Balance Sheet",
    },
    {
        "id": 10,
        "question": "What was the company's exact revenue from automotive sales in Germany?",
        "expected_answer": "I could not find this in the report. (Tests hallucination suppression)",
        "expected_pages": [],
        "category": "Negative Test (Anti-Hallucination)",
    },
]


def run_evaluation(
    model_name: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rag = FinancialRAGChain(model_name=model_name)
    results = []

    print("\n" + "=" * 80)
    print("ANNUAL REPORT Q&A ASSISTANT EVALUATION (10 TEST CASES)")
    print(f"Model: {rag.primary_model_name} (with fallback resilience)")
    if source_filter:
        print(f"Report: {source_filter}")
    print("=" * 80 + "\n")

    for test in TEST_SUITE:
        qid, question = test["id"], test["question"]
        print(f"[{qid}/10] Testing: {question}")

        t0 = time.time()
        res = rag.query(question, source_filter=source_filter)
        duration = round(time.time() - t0, 2)

        answer = res["answer"].strip()
        sources = res["sources"]
        expected_pages = test["expected_pages"]

        if qid == 10:
            # Negative test: the assistant must admit it cannot find the answer
            is_correct = any(
                phrase in answer.lower()
                for phrase in ("could not find", "not found", "not mentioned")
            )
        else:
            citation_match = any(p in sources for p in expected_pages)
            is_correct = citation_match and len(answer) > 10

        status = "✅ PASS" if is_correct else "⚠️ REVIEW"

        results.append({
            "id": qid,
            "category": test["category"],
            "question": question,
            "expected_pages": expected_pages,
            "retrieved_pages": sources,
            "expected_answer": test["expected_answer"],
            "generated_answer": answer,
            "status": status,
            "duration": f"{duration}s",
            "model_used": res["model_used"],
        })
        print(f"       -> {status} | Retrieved Pages: {sources} | Time: {duration}s\n")

    passed = sum(1 for r in results if r["status"].startswith("✅"))
    print(f"Summary: {passed}/{len(results)} passed")
    return results


def print_markdown_table(results: List[Dict[str, Any]]) -> str:
    print("\n" + "#" * 80)
    print("### Evaluation Results Table (Markdown for README)")
    print("#" * 80 + "\n")

    md = "| # | Category | Question | Expected Page | Retrieved Pages | Answer (short) | Status |\n"
    md += "|---|---|---|---|---|---|---|\n"

    for r in results:
        exp = ", ".join(map(str, r["expected_pages"])) if r["expected_pages"] else "None (negative test)"
        ret = ", ".join(map(str, r["retrieved_pages"])) if r["retrieved_pages"] else "None"
        short = r["generated_answer"].replace("\n", " ")
        short = (short[:90] + "...") if len(short) > 90 else short
        short = short.replace("|", "/")
        md += f"| {r['id']} | {r['category']} | {r['question']} | {exp} | {ret} | {short} | {r['status']} |\n"

    print(md)
    return md


if __name__ == "__main__":
    report_filter = sys.argv[1] if len(sys.argv) > 1 else None
    print_markdown_table(run_evaluation(source_filter=report_filter))
