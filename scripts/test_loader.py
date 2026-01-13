from irlib.datasets.mongo_loader import load_collection

if __name__ == "__main__":
    collection_name = "toy_demo"   # ή "CF" ή ό,τι έχεις βάλει στο ingest

    docs, queries, qrels = load_collection(collection_name)

    print(f"[{collection_name}] documents:", len(docs))
    print(f"[{collection_name}] queries:", len(queries))
    print(f"[{collection_name}] qrels:", len(qrels))

    # Δείξε 1-2 για test
    if docs:
        print("\nExample doc:", docs[0]["id"], "->", docs[0]["text"][:80], "...")
    if queries:
        print("Example query:", queries[0]["id"], "->", queries[0]["text"])
    if qrels:
        print("Example qrel:", qrels[0])
