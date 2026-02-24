from irlib.utilities.mongo import get_db

if __name__ == "__main__":
    db = get_db()
    print("Databases:", db.client.list_database_names())
    print("Collections in IR_Lib:", db.list_collection_names())
