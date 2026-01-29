# src/irlib/datasets_insert/mongo_loader.py
from typing import List, Dict, Tuple, Optional

from irlib.utils.mongo import get_db


def load_collection(
    collection_name: str,
    db_name: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Φορτώνει μια IR συλλογή από τη MongoDB.

    Επιστρέφει:
      - documents: [{ "id": str, "text": str }, ...]
      - queries:   [{ "id": str, "text": str }, ...]
      - qrels:     [{ "query_id": str, "doc_id": str, "relevance": int }, ...]

    Η ingest_collection() που φτιάξαμε πριν πρέπει να έχει
    δημιουργήσει τα docs/queries/qrels με αυτό το schema.
    """
    db = get_db(db_name) if db_name else get_db()

    docs_col = db["Documents"]
    queries_col = db["Queries"]
    qrels_col = db["Qrels"]

    # Φέρνουμε μόνο τα στοιχεία αυτής της συλλογής
    docs_cursor = docs_col.find({"collection": collection_name})
    queries_cursor = queries_col.find({"collection": collection_name})
    qrels_cursor = qrels_col.find({"collection": collection_name})

    documents: List[Dict] = [
        {"id": str(doc["_id"]), "text": doc["text"]}
        for doc in docs_cursor
    ]

    queries: List[Dict] = [
        {"id": str(q["_id"]), "text": q["text"]}
        for q in queries_cursor
    ]

    qrels: List[Dict] = [
        {
            "query_id": str(qr["query_id"]),
            "doc_id": str(qr["doc_id"]),
            "relevance": int(qr.get("relevance", 1)),
        }
        for qr in qrels_cursor
    ]

    return documents, queries, qrels
