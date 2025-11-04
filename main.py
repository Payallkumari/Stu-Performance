# main.py
import os
from mongodb_import import load_csv_to_mongo, avg_math_by_parent_education
from analysis_ml import load_data, task2_avg_by_gender, task3_classification, task4_kmeans, elbow_method, task5_regression

def main():
    print("=== Step 1: Load CSV into MongoDB and run aggregation ===")
    coll = load_csv_to_mongo()
    agg = avg_math_by_parent_education(coll)
    for r in agg:
        print(f"{r['_id']}: avg_math={r['avg_math']:.2f} (n={r['count']})")

    print("\n=== Step 2: Pandas analysis & plots ===")
    df = load_data()
    avg = task2_avg_by_gender(df)

    print("\n=== Step 3: Train SVM classifier (Pass/Fail) ===")
    model, cm, acc = task3_classification(df)

    print("\n=== Step 4: KMeans clustering & Elbow method ===")
    kmeans, df_clusters = task4_kmeans(df, n_clusters=3)
    ks, inertias = elbow_method(df, k_max=8)

    print("\n=== Step 5: Linear regression (writing ~ reading) ===")
    reg, r2 = task5_regression(df)

    print("\nAll done. Plots in ./plots/ .")

if __name__ == "__main__":
    main()
