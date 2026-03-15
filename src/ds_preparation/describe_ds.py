import pandas as pd

INPUT_DATASET_PATH = "data/dataset_incomplete.csv"
COLUMN_NAME = "query"

df = pd.read_csv(INPUT_DATASET_PATH)
query_type_count = {}
total_rows = len(df)

for row_idx, row in df.iterrows():
    print(f"Processing row ({row_idx + 1} / {total_rows}) ...")
    query_type = row[COLUMN_NAME].split()[0]
    print(f"Row {row_idx + 1} query type: '{query_type}'")

    if query_type not in query_type_count:
        query_type_count[query_type] = 0
    else:
        query_type_count[query_type] += 1

print(f"All {total_rows} rows processed successfully")

print("******************** DATASET SUMMARY ********************")
for q_type, q_count in query_type_count.items():
    print(f"\t- {q_type}: {q_count}")
print("*********************************************************")