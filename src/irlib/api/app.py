# src/irlib/api/app.py
"""
Flask API για το IR Benchmarking framework.

Εκκίνηση:
    python src/irlib/api/app.py

Endpoints:
    GET  /collections       → διαθέσιμες συλλογές στη MongoDB
    GET  /models            → διαθέσιμα μοντέλα
    POST /run               → τρέχει ένα μοντέλο
    POST /compare           → τρέχει πολλά μοντέλα και συγκρίνει
    GET  /results           → αποθηκευμένα αποτελέσματα από τη MongoDB
"""

import sys
import time
from pathlib import Path

# Path setup — ώστε να βρίσκονται models, utilities, Preprocess
_API_DIR = Path(__file__).resolve().parent
_IRLIB_DIR = _API_DIR.parent
sys.path.insert(0, str(_IRLIB_DIR))

from flask import Flask, request, jsonify
from numpy import mean, std

from irlib.collection_builder import build_collection_from_mongo
from irlib.utilities.mongo import get_db
from irlib.api.registry import get_model_class, list_models

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_single(model_name: str, collection_name: str,
                runs: int, k, stopwords: bool, min_freq: int) -> dict:
    """Τρέχει ένα μοντέλο N φορές και επιστρέφει αποτελέσματα."""

    col = build_collection_from_mongo(collection_name)
    ModelClass = get_model_class(model_name)

    map_scores = []
    all_precision = []
    all_recall = []
    total_start = time.time()

    for i in range(runs):
        model = ModelClass(col)
        model.fit(min_freq=min_freq, stopwords=stopwords)
        model.evaluate(k=k)
        map_scores.append(float(mean(model.precision)))
        all_precision.append([round(float(p), 6) for p in model.precision])
        all_recall.append([round(float(r), 6) for r in model.recall])

    elapsed = round(time.time() - total_start, 2)

    return {
        "model":       model_name,
        "collection":  collection_name,
        "runs":        runs,
        "map_mean":    round(float(mean(map_scores)), 6),
        "map_std":     round(float(std(map_scores)), 6),
        "map_per_run": [round(s, 6) for s in map_scores],
        "precision":   all_precision,
        "recall":      all_recall,
        "elapsed_sec": elapsed,
    }


def _save_result(result: dict):
    """Αποθηκεύει αποτέλεσμα στη MongoDB."""
    db = get_db()
    db["Results"].insert_one({
        **result,
        "timestamp": time.time(),
    })


# ---------------------------------------------------------------------------
# GET /collections
# ---------------------------------------------------------------------------

@app.route("/collections", methods=["GET"])
def get_collections():
    """Επιστρέφει τις συλλογές που υπάρχουν στη MongoDB."""
    db = get_db()
    collections = db["Documents"].distinct("collection")
    return jsonify({"collections": collections})


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@app.route("/models", methods=["GET"])
def get_models():
    """Επιστρέφει τα διαθέσιμα μοντέλα από το registry."""
    return jsonify({"models": list_models()})


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------

@app.route("/run", methods=["POST"])
def run_model():
    """
    Τρέχει ένα μοντέλο σε μια συλλογή.

    Body (JSON):
        model      (str)  : π.χ. "GSB", "BM25"
        collection (str)  : π.χ. "CF"
        runs       (int)  : αριθμός επαναλήψεων (default: 1)
        k          (int)  : cutoff (default: null = όλα τα docs)
        stopwords  (bool) : (default: true)
        min_freq   (int)  : για apriori (default: 1)
        save       (bool) : αποθήκευση στη MongoDB (default: false)
    """
    data = request.get_json()

    # Validation
    if not data or "model" not in data or "collection" not in data:
        return jsonify({"error": "Απαιτούνται τα πεδία 'model' και 'collection'"}), 400

    model_name      = data["model"].upper()
    collection_name = data["collection"].upper()
    runs            = int(data.get("runs", 1))
    k               = data.get("k", None)
    stopwords       = bool(data.get("stopwords", True))
    min_freq        = int(data.get("min_freq", 1))
    save            = bool(data.get("save", False))

    try:
        result = _run_single(model_name, collection_name,
                             runs, k, stopwords, min_freq)
        if save:
            _save_result(result)
        return jsonify(result)
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------

@app.route("/compare", methods=["POST"])
def compare_models():
    """
    Τρέχει πολλά μοντέλα στην ίδια συλλογή και επιστρέφει σύγκριση.

    Body (JSON):
        models     (list) : π.χ. ["GSB", "BM25"]
        collection (str)  : π.χ. "CF"
        runs       (int)  : (default: 1)
        k          (int)  : (default: null)
        stopwords  (bool) : (default: true)
        save       (bool) : (default: false)
    """
    data = request.get_json()

    if not data or "models" not in data or "collection" not in data:
        return jsonify({"error": "Απαιτούνται τα πεδία 'models' και 'collection'"}), 400

    models_list     = [m.upper() for m in data["models"]]
    collection_name = data["collection"].upper()
    runs            = int(data.get("runs", 1))
    k               = data.get("k", None)
    stopwords       = bool(data.get("stopwords", True))
    min_freq        = int(data.get("min_freq", 1))
    save            = bool(data.get("save", False))

    results = {}
    errors  = {}

    for model_name in models_list:
        try:
            r = _run_single(model_name, collection_name,
                            runs, k, stopwords, min_freq)
            results[model_name] = r
            if save:
                _save_result(r)
        except Exception as e:
            errors[model_name] = str(e)

    return jsonify({
        "collection": collection_name,
        "results":    results,
        "errors":     errors if errors else None,
    })


# ---------------------------------------------------------------------------
# GET /results
# ---------------------------------------------------------------------------

@app.route("/results", methods=["GET"])
def get_results():
    """
    Επιστρέφει αποθηκευμένα αποτελέσματα από τη MongoDB.

    Query params:
        model      : φιλτράρισμα ανά μοντέλο
        collection : φιλτράρισμα ανά συλλογή
        limit      : max αποτελέσματα (default: 20)
    """
    db = get_db()

    query = {}
    if request.args.get("model"):
        query["model"] = request.args["model"].upper()
    if request.args.get("collection"):
        query["collection"] = request.args["collection"].upper()

    limit = int(request.args.get("limit", 20))

    cursor = db["Results"].find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
    results = list(cursor)

    return jsonify({"results": results, "count": len(results)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting IR Benchmarking API...")
    print(f"Models available: {list_models()}")
    app.run(debug=True, host="0.0.0.0", port=5000)