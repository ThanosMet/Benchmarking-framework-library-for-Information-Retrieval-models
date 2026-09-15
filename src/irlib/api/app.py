# src/irlib/api/app.py
"""
Flask API για το IR Benchmarking framework.

Εκκίνηση:
    python src/irlib/api/app.py

    powershell command example: Invoke-WebRequest -Uri "http://127.0.0.1:5000/run" -Method POST -ContentType "application/json" -Body '{"model": "BM25", "collection": "CF", "runs": 1}'

Endpoints:
    GET  /collections       → διαθέσιμες συλλογές στη MongoDB
    GET  /models            → διαθέσιμα μοντέλα
    GET  /model_params      → παράμετροι ανά μοντέλο
    POST /run               → τρέχει ένα μοντέλο
    POST /compare           → τρέχει πολλά μοντέλα και συγκρίνει
    GET  /results           → αποθηκευμένα αποτελέσματα από τη MongoDB
"""

import sys
import time
import traceback
import ast
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

# Μοντέλα που δέχονται window parameter
_WINDOWED_MODELS = {"WINDOWEDGSB"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(model_name: str, col, extra_params: dict):
    """Αρχικοποιεί το σωστό μοντέλο με τις κατάλληλες παραμέτρους."""
    ModelClass = get_model_class(model_name)

    # 1. Parse Window safely (Handle Streamlit's 8.0 float conversion)
    window_raw = extra_params.get("window", 8)
    try:
        window = float(window_raw)
        # If it's 1.0 or higher, it represents a fixed number of words (Integer)
        # If it's less than 1.0 (e.g. 0.5), it represents a percentage (Float)
        if window >= 1.0:
            window = int(window)
    except (ValueError, TypeError):
        window = 8

    # 2. Parse Clusters safely
    clusters = int(extra_params.get("clusters", 5))

    # 3. Parse Condition safely
    condition_raw = extra_params.get("condition", {})

    if isinstance(condition_raw, dict):
        condition_dict = condition_raw
    elif isinstance(condition_raw, str) and condition_raw.strip() != "":
        try:
            condition_dict = ast.literal_eval(condition_raw)
        except Exception:
            condition_dict = {}
    else:
        condition_dict = {}

    # 4. Route to the correct constructor based on what the model requires
    if model_name in ["WINDOWEDGSB", "GSBWINDOW"]:
        return ModelClass(col, window=window)

    elif model_name == "PGSB":
        return ModelClass(col, clusters=clusters, condition=condition_dict)

    elif model_name == "PGSBW":
        return ModelClass(col, window=window, clusters=clusters, condition=condition_dict)

    elif model_name == "CONGSB":
        # ConGSB also accepts **kwargs for cluster_optimization
        return ModelClass(col, clusters=clusters, cond=condition_dict)

    elif model_name == "CONGSBW":
        return ModelClass(col, window=window, clusters=clusters, condition=condition_dict)


    elif model_name == "PYLATE":
        pretrained_model = str(extra_params.get("pretrained_model", "lightonai/colbertv2.0"))
        return ModelClass(col, pretrained_model=pretrained_model)

    # Default for base models like BM25, GSB
    return ModelClass(col)

"""Τρέχει ένα μοντέλο N φορές και επιστρέφει αποτελέσματα."""
def _run_single(model_name: str, collection_name: str,
                runs: int, k, stopwords: bool,
                min_freq: int, extra_params: dict) -> dict:

    col = build_collection_from_mongo(collection_name)

    map_scores = []

    all_precision = []
    all_recall = []
    all_average_precision = []
    all_mrr = []

    total_start = time.time()

    for i in range(runs):
        model = _build_model(model_name, col, extra_params)

        model.fit(
            min_freq=min_freq,
            stopwords=stopwords
        )

        model.evaluate(k=k)

        # MAP = mean Average Precision over all queries
        run_map = float(mean(model.average_precision))
        map_scores.append(run_map)

        # Per-query metrics
        all_precision.append(
            [round(float(p), 6) for p in model.precision]
        )

        all_recall.append(
            [round(float(r), 6) for r in model.recall]
        )

        all_average_precision.append(
            [round(float(ap), 6) for ap in model.average_precision]
        )

        all_mrr.append(
            [round(float(rr), 6) for rr in model.mrr]
        )

    elapsed = round(time.time() - total_start, 2)

    return {
        "model": model_name,
        "collection": collection_name,
        "runs": runs,

        "map_mean": round(float(mean(map_scores)), 6),
        "map_std": round(float(std(map_scores)), 6),
        "map_per_run": [round(s, 6) for s in map_scores],

        "precision": all_precision,
        "recall": all_recall,
        "average_precision": all_average_precision,
        "mrr": all_mrr,

        "elapsed_sec": elapsed,
        "params": extra_params,
    }


def _save_result(result: dict):
    """Αποθηκεύει αποτέλεσμα στη MongoDB."""
    db = get_db()
    db["Results"].insert_one({**result, "timestamp": time.time()})


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
# GET /model_params  — ποιες extra παράμετροι δέχεται κάθε μοντέλο
# ---------------------------------------------------------------------------

@app.route("/model_params", methods=["GET"])
def get_model_params():
    window_param = {"name": "window", "type": "number", "default": 8, "help": "Window size (int/float)"}
    clusters_param = {"name": "clusters", "type": "number", "default": 5, "help": "Number of clusters"}
    cond_param = {"name": "condition", "type": "string", "default": "{}", "help": "JSON string eg: {'edge': 0.5}"}
    pylate_param = {"name": "pretrained_model", "type": "string", "default": "lightonai/colbertv2.0",
                    "help": "HuggingFace model ID (e.g., lightonai/colbertv2.0)"}

    params = {
        "GSB": [],
        "BM25": [],
        "GOW": [],
        "TFIDF": [],
        "SBERT": [],
        "LSI": [],
        "WINDOWEDGSB": [window_param],
        "GSBWINDOW": [window_param],
        "PGSB": [clusters_param, cond_param],
        "PGSBW": [window_param, clusters_param, cond_param],
        "CONGSB": [clusters_param, cond_param],
        "CONGSBW": [window_param, clusters_param, cond_param],
        "PYLATE": [pylate_param],
    }
    return jsonify(params)


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
    k = data.get("k", 10)

    if k is None:
        k = 10
    else:
        k = int(k)
    stopwords       = bool(data.get("stopwords", True))
    min_freq        = int(data.get("min_freq", 1))
    save            = bool(data.get("save", False))
    extra_params    = data.get("params", {})

    try:
        result = _run_single(model_name, collection_name,
                             runs, k, stopwords, min_freq, extra_params)
        if save:
            _save_result(result)
        return jsonify(result)
    except KeyError as e:
        print("KEY ERROR IN /run:", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": f"KeyError: {str(e)}"}), 400
    except Exception as e:
        print("CRASH IN /run:", file=sys.stderr)
        traceback.print_exc()
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
    k = data.get("k", 10)

    if k is None:
        k = 10
    else:
        k = int(k)
    stopwords       = bool(data.get("stopwords", True))
    min_freq        = int(data.get("min_freq", 1))
    save            = bool(data.get("save", False))
    extra_params    = data.get("params", {})

    results = {}
    errors  = {}

    for model_name in models_list:
        try:
            r = _run_single(model_name, collection_name,
                            runs, k, stopwords, min_freq, extra_params)
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
# POST /results/delete (Διαγραφή συγκεκριμένου αποτελέσματος)
# ---------------------------------------------------------------------------

@app.route("/results/delete", methods=["POST"])
def delete_specific_result():
    data = request.get_json()
    timestamp = data.get("timestamp")
    if not timestamp:
        return jsonify({"error": "Missing timestamp"}), 400

    db = get_db()
    # Διαγράφει μόνο το συγκεκριμένο αποτέλεσμα που έχει αυτό το timestamp
    res = db["Results"].delete_one({"timestamp": float(timestamp)})
    return jsonify({"success": True, "deleted_count": res.deleted_count})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting IR Benchmarking API...")
    print(f"Models available: {list_models()}")
    app.run(debug=True, host="0.0.0.0", port=5000)
