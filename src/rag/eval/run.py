from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the shared RAG mini-benchmark.")
    parser.add_argument("--embedder", choices=["hashing", "bge-m3"], default="hashing")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args(argv)

    summary = run_benchmark(
        embedder_name=args.embedder,
        top_k=args.top_k,
        use_llm=args.use_llm,
    )

    header = f"{'qid':<4} {'lang':<4} {'case':<16} {'hit':<5} {'recall':<7} {'top':<7} retrieved"
    print(header)
    print("-" * len(header))
    for row in summary["rows"]:
        hit = "-" if row["hit"] is None else ("Y" if row["hit"] else "N")
        recall = "-" if row["recall"] is None else f"{row['recall']:.2f}"
        print(
            f"{row['qid']:<4} {row['language']:<4} {row['case_type']:<16} "
            f"{hit:<5} {recall:<7} {row['top_score']:<7} {','.join(row['retrieved'])}"
        )

    print()
    print(f"embedder={summary['embedder']}  llm={summary['llm']}  top_k={summary['top_k']}")
    print(
        f"questions={summary['questions']}  scored={summary['scored_questions']}  "
        f"hit@k={summary['hit_at_k']}  recall@k={summary['recall_at_k']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
