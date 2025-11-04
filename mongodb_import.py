# mongodb_import.py
import os
import pandas as pd
from pymongo import MongoClient



# Configuration
DATA_PATH = os.path.join("data", "StudentsPerformance.csv")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "student_db"
COLLECTION_NAME = "students"

def load_csv_to_mongo(csv_path=DATA_PATH, mongo_uri=MONGO_URI):
    df = pd.read_csv(csv_path)
    # convert column names to match CSV exactly (no changes)
    client = MongoClient(mongo_uri)
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]

    # Optional: drop existing collection to start fresh
    coll.drop()
    # Insert records
    records = df.to_dict(orient="records")
    coll.insert_many(records)
    print(f"Inserted {len(records)} records into {DB_NAME}.{COLLECTION_NAME}")
    return coll

def avg_math_by_parent_education(collection):
    pipeline = [
        {"$group": {
            "_id": "$parental level of education",
            "avg_math": {"$avg": "$math score"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"avg_math": -1}}
    ]
    result = list(collection.aggregate(pipeline))
    return result

if __name__ == "__main__":
    coll = load_csv_to_mongo()
    agg = avg_math_by_parent_education(coll)
    print("Average math score by parental level of education:")
    for r in agg:
        print(f"{r['_id']}: avg_math={r['avg_math']:.2f} (n={r['count']})")
