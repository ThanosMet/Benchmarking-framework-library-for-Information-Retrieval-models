# scripts/run_gsb.py
"""
Αυτοματοποιημένο GSB testing με MongoDB.
Αντικαθιστά το GSB_testing.py — χωρίς file I/O, χωρίς paths, χωρίς expir_start().

Χρήση:
    python scripts/run_gsb.py                        # τρέχει CF με defaults
    python scripts/run_gsb.py --collection NPL       # άλλη συλλογή
    python scripts/run_gsb.py --runs 3 --k 20        # custom παράμετροι
"""

import sys
from pathlib import Path
import argparse
import os
from numpy import mean
from pandas import DataFrame

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "irlib"))

from models.GSB import GSBModel
from utilities.Result_handling import res_to_excel, write
from irlib.collection_builder import build_collection_from_mongo




def parse_args():
    parser = argparse.ArgumentParser(description="GSB Model Benchmarking via MongoDB")
    parser.add_argument("--collection", type=str,   default="CF",
                        help="Όνομα συλλογής στη MongoDB (default: CF)")
    parser.add_argument("--runs",       type=int,   default=5,
                        help="Πλήθος επαναλήψεων για στατιστική αξιοπιστία (default: 5)")
    parser.add_argument("--k",          type=int,   default=None,
                        help="Cutoff k για precision/recall (default: όλα τα docs)")
    parser.add_argument("--min_freq",   type=int,   default=1,
                        help="Ελάχιστη συχνότητα για apriori termsets (default: 1)")
    parser.add_argument("--stopwords",  action="store_true", default=True,
                        help="Εφαρμογή stopwords κατά το fit (default: True)")
    parser.add_argument("--dest",       type=str,   default="results",
                        help="Φάκελος για αποθήκευση αποτελεσμάτων (default: results/)")
    parser.add_argument("--k_core",     action="store_true", default=False,
                        help="Ενεργοποίηση k-core pruning στο GSBModel (default: False)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Δημιουργία φακέλου αποτελεσμάτων αν δεν υπάρχει ---
    os.makedirs(args.dest, exist_ok=True)
    results_file = f"[{args.collection}]GSBTesting.xlsx"

    print("=" * 60)
    print(f"  GSB Benchmarking — Collection: {args.collection}")
    print(f"  Runs: {args.runs} | k: {args.k} | min_freq: {args.min_freq}")
    print(f"  Stopwords: {args.stopwords} | k-core: {args.k_core}")
    print(f"  Output: {args.dest}/{results_file}")
    print("=" * 60)

    # --- Φόρτωση συλλογής από MongoDB (1 φορά, reuse σε όλα τα runs) ---
    col = build_collection_from_mongo(args.collection)
    print(f"\nCollection: {col.num_docs} max_doc_id, "
          f"{len(col.queries)} queries, "
          f"{len(col.inverted_index)} vocab terms\n")

    # --- Επαναλήψεις ---
    map_scores = []
    run_names = []

    for i in range(args.runs):
        print(f"\n{'─'*40}")
        print(f"  Run {i + 1} / {args.runs}")
        print(f"{'─'*40}")

        model = GSBModel(col, k_core_bool=args.k_core)
        model.fit(min_freq=args.min_freq, stopwords=args.stopwords)
        model.evaluate(k=args.k)

        map_score = mean(model.precision)
        map_scores.append(map_score)
        run_name = f"run_{i}"
        run_names.append(run_name)

        print(f"\n  → MAP = {map_score:.4f}")

        # Αποθήκευση per-run αποτελεσμάτων
        res_to_excel(model, results_file, args.dest, sheetname=run_name)

    # --- Aggregate sheet ---
    print(f"\n{'=' * 60}")
    print(f"  Αποτελέσματα {args.collection} — {args.runs} runs")
    print(f"{'=' * 60}")
    for name, score in zip(run_names, map_scores):
        print(f"  {name}: MAP = {score:.4f}")
    print(f"  {'─'*30}")
    print(f"  Mean MAP: {mean(map_scores):.4f}")
    print(f"  Std  MAP: {__import__('numpy').std(map_scores):.4f}")

    df = DataFrame(list(zip(run_names, map_scores)), columns=["run", "MAP"])
    write(xl_namefile=results_file, dest_path=args.dest, sheetname="aggregate", data=df)

    print(f"\n  Αποτελέσματα αποθηκεύτηκαν → {args.dest}/{results_file}")


if __name__ == "__main__":
    main()