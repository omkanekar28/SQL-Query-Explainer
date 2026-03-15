import os
import time
import pandas as pd
from llama_cpp import Llama
from src.common import (
    PromptManager, OUTPUT_DF_DICT_TEMPLATE, SQL_EXPLAINER_SYSTEM_PROMPT_DS_PREP,
    LlamaCppDatasetPreparationModelSettings,  LlamaCppDatasetPreparationInferenceSettings, 
    separate_thinking_and_response
)

INPUT_DATASET_FILEPATH = "data/dataset_incomplete.csv"
INPUT_DATASET_SYSTEM_COLUMN = "system"
INPUT_DATASET_QUERY_COLUMN = "query"
INPUT_DATASET_EXPLANATION_COLUMN = "explanation"
INPUT_DATASET_REASONING_COLUMN = "reasoning"

OUTPUT_DATASET_FILEPATH = "data/dataset_complete.csv"

GGUF_MODEL_FILEPATH = "models/base/Qwen_Qwen3-4B-Q5_K_M.gguf"

output_df_dict = OUTPUT_DF_DICT_TEMPLATE.copy()

model_loading_start_time = time.time()
print(f"Loading model '{GGUF_MODEL_FILEPATH}' ...")
model = Llama(
    model_path=GGUF_MODEL_FILEPATH,
    n_ctx=LlamaCppDatasetPreparationModelSettings.N_CTX,
    n_gpu_layers=LlamaCppDatasetPreparationModelSettings.N_GPU_LAYERS,
    n_batch=LlamaCppDatasetPreparationModelSettings.N_BATCH,
    n_threads=LlamaCppDatasetPreparationModelSettings.N_THREADS,
    n_threads_batch=LlamaCppDatasetPreparationModelSettings.N_THREADS_BATCH,
    verbose=LlamaCppDatasetPreparationModelSettings.VERBOSE
)
print(f"Model '{GGUF_MODEL_FILEPATH}' loaded successfully in {time.time() - model_loading_start_time:.2f} seconds")

prompt_manager = PromptManager(SQL_EXPLAINER_SYSTEM_PROMPT_DS_PREP)

input_df = pd.read_csv(INPUT_DATASET_FILEPATH)

# Resume from checkpoint if output file already exists
if os.path.exists(OUTPUT_DATASET_FILEPATH):
    existing_df = pd.read_csv(OUTPUT_DATASET_FILEPATH)
    rows_already_processed = len(existing_df)
    output_df_dict = existing_df.to_dict(orient="list")
    input_df = input_df.iloc[rows_already_processed:]
    print(f"Resuming from row {rows_already_processed + 1} ({len(input_df)} rows remaining) ...")
else:
    rows_already_processed = 0

total_processing_start_time = time.time()
print(f"Performing dataset preparation on {len(input_df)} rows ...")
for row_idx, row in input_df.iterrows():
    row_processing_start_time = time.time()
    print(f"\n\nProcessing row ({row_idx + 1} / {len(input_df)}) ...\n\n")
    
    output_df_dict[INPUT_DATASET_SYSTEM_COLUMN].append(row[INPUT_DATASET_SYSTEM_COLUMN])
    output_df_dict[INPUT_DATASET_QUERY_COLUMN].append(row[INPUT_DATASET_QUERY_COLUMN])
    
    try:
        row_query = row[INPUT_DATASET_QUERY_COLUMN]
        messages = prompt_manager.get_prompt_messages(query=row_query)
        print(f"QUERY:- {row_query}")
        
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=LlamaCppDatasetPreparationInferenceSettings.MAX_TOKENS,
            temperature=LlamaCppDatasetPreparationInferenceSettings.TEMPERATURE,
            presence_penalty=LlamaCppDatasetPreparationInferenceSettings.PRESENCE_PENALTY,
            repeat_penalty=LlamaCppDatasetPreparationInferenceSettings.REPEAT_PENALTY,
            top_k=LlamaCppDatasetPreparationInferenceSettings.TOP_K,
            top_p=LlamaCppDatasetPreparationInferenceSettings.TOP_P,
        )
        response_text = response["choices"][0]["message"]["content"]
        reasoning, explanation = separate_thinking_and_response(response_text)
        usage = response["usage"]
        print(explanation)
        print(usage)
        
        # Saving output df after every row
        output_df_dict[INPUT_DATASET_EXPLANATION_COLUMN].append(explanation)
        output_df_dict[INPUT_DATASET_REASONING_COLUMN].append(reasoning)
        output_df = pd.DataFrame(output_df_dict)
        output_df.to_csv(OUTPUT_DATASET_FILEPATH, index=False)
        
        print(f"Row {row_idx + 1} processed in {time.time() - row_processing_start_time:.2f} seconds")
    except Exception as e:
        print(f"[WARNING] Failed to process row {row_idx + 1}! {str(e)}")

        # Saving output df after every row
        output_df_dict[INPUT_DATASET_EXPLANATION_COLUMN].append(None)
        output_df_dict[INPUT_DATASET_REASONING_COLUMN].append(None)
        output_df = pd.DataFrame(output_df_dict)
        output_df.to_csv(OUTPUT_DATASET_FILEPATH, index=False)


print(f"Dataset preparation for {len(input_df)} rows finished in {time.time() - total_processing_start_time:.2f} seconds")