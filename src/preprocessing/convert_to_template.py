import os
import re
import pandas as pd
from src.common import OUTPUT_DF_DICT_TEMPLATE
from src.common import PromptManager, SQL_EXPLAINER_SYSTEM_PROMPT_DS_PREP

INPUT_DATASET_PATH = "data/dataset-for-developing-natural-language-interface"
COLUMN_NAME = "sql"
INNER_DICT_COLUMN_NAME = "human_readable"
OUTPUT_DF_DICT = OUTPUT_DF_DICT_TEMPLATE.copy()
OUTPUT_CSV_SAVE_PATH = "data/dataset_incomplete.csv"

prompt_manager = PromptManager(system=SQL_EXPLAINER_SYSTEM_PROMPT_DS_PREP)

files = os.listdir(INPUT_DATASET_PATH)
no_of_files = len(files)

failed_files_count = 0
failed_rows_count = 0

for file_idx, filename in enumerate(files):
    
    try:
        print(f"Processing file '{filename}' ({file_idx + 1} / {no_of_files}) ...")
        
        # Reading the file
        filepath = os.path.join(INPUT_DATASET_PATH, filename)
        input_df = pd.read_csv(filepath)
        no_of_rows = len(input_df)

        for row_idx, row in input_df.iterrows():
            
            try:
                print(f"Processing row {row_idx + 1} / {no_of_rows} ...")

                # Fetching the SQL query
                match = re.search(rf"'{INNER_DICT_COLUMN_NAME}':\s*'([^']+)'", row[COLUMN_NAME])
                if not match:
                    raise ValueError(f"Could not extract '{INNER_DICT_COLUMN_NAME}' from row {row_idx}")
                query = match.group(1)

                # Updating the output dataframe
                OUTPUT_DF_DICT["system"].append(prompt_manager.SYSTEM_PROMPT)
                OUTPUT_DF_DICT["query"].append(query)
                OUTPUT_DF_DICT["explanation"].append("")
                OUTPUT_DF_DICT["reasoning"].append("")
            
            except Exception as e:
                print(
                    f"[WARNING] Failed to process row {row_idx + 1}! "
                    f"{str(e)}. "
                    f"Skipping ..."
                )
                failed_rows_count += 1
        
        print(f"File '{filename}' ({file_idx + 1} / {no_of_files}) processed successfully")

    except Exception as e:
        print(
            f"[WARNING] Failed to process file '{filename}' ({file_idx + 1} / {no_of_files})! ", 
            f"{str(e)}. ", 
            f"Skipping ..."
        )
        failed_files_count += 1

print(f"All {no_of_files} files processed successfully")
OUTPUT_DF = pd.DataFrame(OUTPUT_DF_DICT)
OUTPUT_DF.to_csv(OUTPUT_CSV_SAVE_PATH, index=False)
print(f"Final dataset saved successfully at '{OUTPUT_CSV_SAVE_PATH}'")

print("******************** CONVERSION FINISHED ********************")
print(f"\t- No of files processed: {no_of_files - failed_files_count}")
print(f"\t- No of files failed: {failed_files_count}")
print(f"\t- No of rows processed: {len(OUTPUT_DF_DICT['query']) - failed_rows_count}")
print(f"\t- No of rows failed: {failed_rows_count}")
print("*************************************************************")