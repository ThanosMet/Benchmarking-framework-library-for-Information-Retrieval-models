# scripts/run_bm25.py
"""
Αυτοματοποιημένο BM25 testing με MongoDB.

Χρήση:
    python scripts/run_bm25.py                        # τρέχει CF με defaults
    python scripts/run_bm25.py --collection NPL       # άλλη συλλογή
    python scripts/run_bm25.py --runs 3 --k 20        # custom παράμετροι
"""

import argparse
import os
import sys
from pathlib import Path
from numpy import mean, std
from pandas import DataFrame


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "irlib"))


from models.BM25 import BM25Model
from utilities.Result_handling import res_to_excel, write
from irlib.collection_builder import build_collection_from_mongo


def parse_args():
    parser = argparse.ArgumentParser(description="BM25 Model Benchmarking via MongoDB")
    parser.add_argument("--collection", type=str, default="CF",
                        help="Όνομα συλλογής στη MongoDB (default: CF)")
    parser.add_argument("--runs",       type=int, default=5,
                        help="Πλήθος επαναλήψεων (default: 5)")
    parser.add_argument("--k",          type=int, default=None,
                        help="Cutoff k για precision/recall (default: όλα τα docs)")
    parser.add_argument("--stopwords",  action="store_true", default=True,
                        help="Εφαρμογή stopwords στα queries (default: True)")
    parser.add_argument("--dest",       type=str, default="results",
                        help="Φάκελος για αποθήκευση αποτελεσμάτων (default: results/)")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.dest, exist_ok=True)
    results_file = f"[{args.collection}]BM25Testing.xlsx"

    print("=" * 60)
    print(f"  BM25 Benchmarking — Collection: {args.collection}")
    print(f"  Runs: {args.runs} | k: {args.k} | Stopwords: {args.stopwords}")
    print(f"  Output: {args.dest}/{results_file}")
    print("=" * 60)

    # Φόρτωση συλλογής από MongoDB (1 φορά)
    col = build_collection_from_mongo(args.collection)
    print(f"\nCollection: {col.num_docs} max_doc_id, "
          f"{len(col.queries)} queries, "
          f"{len(col.inverted_index)} vocab terms\n")

    map_scores = []
    run_names = []

    for i in range(args.runs):
        print(f"\n{'─'*40}")
        print(f"  Run {i + 1} / {args.runs}")
        print(f"{'─'*40}")

        model = BM25Model(col)
        model.fit(stopwords=args.stopwords)
        model.evaluate(k=args.k)

        map_score = mean(model.precision)
        map_scores.append(map_score)
        run_name = f"run_{i}"
        run_names.append(run_name)

        print(f"\n  → MAP = {map_score:.4f}")

        res_to_excel(model, results_file, args.dest, sheetname=run_name)

    # Aggregate
    print(f"\n{'=' * 60}")
    print(f"  Αποτελέσματα {args.collection} — {args.runs} runs")
    print(f"{'=' * 60}")
    for name, score in zip(run_names, map_scores):
        print(f"  {name}: MAP = {score:.4f}")
    print(f"  {'─'*30}")
    print(f"  Mean MAP: {mean(map_scores):.4f}")
    print(f"  Std  MAP: {std(map_scores):.4f}")

    df = DataFrame(list(zip(run_names, map_scores)), columns=["run", "MAP"])
    write(xl_namefile=results_file, dest_path=args.dest, sheetname="aggregate", data=df)

    print(f"\n  Αποτελέσματα αποθηκεύτηκαν → {args.dest}/{results_file}")


if __name__ == "__main__":
    main()