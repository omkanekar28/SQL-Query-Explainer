import time
import pandas as pd
from llama_cpp import Llama
from src.common import LlamaCppDatasetPreparationModelSettings, LlamaCppDatasetPreparationInferenceSettings
from src.common import PromptManager, SQL_EXPLAINER_SYSTEM_PROMPT_DS_PREP

TEST_FILEPATH = "data/splits/test.csv"
QUERY_COLUMN_NAME = "query"

GGUF_MODEL_FILEPATH = "models/base/gemma-3-270m-it-Q5_K_M.gguf"

NO_OF_ROWS_TO_TEST = 10

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

test_df = pd.read_csv(TEST_FILEPATH)
test_column = test_df[QUERY_COLUMN_NAME]

if NO_OF_ROWS_TO_TEST:
    test_column = test_column[:NO_OF_ROWS_TO_TEST]

prompt_manager = PromptManager(SQL_EXPLAINER_SYSTEM_PROMPT_DS_PREP)

total_processing_start_time = time.time()
print(f"Performing zero-shot testing on {NO_OF_ROWS_TO_TEST} testing rows ...")
for row_idx, row_query in enumerate(test_column):
    row_processing_start_time = time.time()
    print(f"\n\nProcessing row ({row_idx + 1} / {NO_OF_ROWS_TO_TEST}) ...\n\n")
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
    explanation = response["choices"][0]["message"]["content"]
    usage = response["usage"]
    print(explanation)
    print(usage)
    print(f"Row {row_idx + 1} processed in {time.time() - row_processing_start_time:.2f} seconds")
print(f"Zero-shot testing for {NO_OF_ROWS_TO_TEST} rows finished in {time.time() - total_processing_start_time:.2f} seconds")