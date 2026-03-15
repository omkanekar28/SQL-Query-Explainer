class DatasetSettings:
    RANDOM_SEED: float = 42

    TRAIN_SPLIT_SIZE: float = 0.9
    VAL_SPLIT_SIZE: float = 0.05
    TEST_SPLIT_SIZE: float = 0.05

# --------------------
# DS Preparation Model
# --------------------
class LlamaCppDatasetPreparationModelSettings:
    N_CTX: int = 6000
    N_GPU_LAYERS: int = -1
    N_BATCH: int = 512
    N_THREADS: int = 8
    N_THREADS_BATCH: int = 8
    VERBOSE: bool = True
    FLAST_ATTN: bool = True


class LlamaCppDatasetPreparationInferenceSettings:
    MAX_TOKENS: int = 4096    # ACCOUNT FOR REASONING
    TEMPERATURE: float = 0.7
    PRESENCE_PENALTY: float = 0.0
    REPEAT_PENALTY: float = 1.1
    TOP_K: int = 20
    TOP_P: float = 0.95


# ---------------
# Finetuned Model
# ---------------
class LlamaCppTrainedModelSettings:
    N_CTX: int = 4096
    N_GPU_LAYERS: int = -1
    N_BATCH: int = 512
    N_THREADS: int = 8
    N_THREADS_BATCH: int = 8
    VERBOSE: bool = True
    FLAST_ATTN: bool = False    # MOSTLY RUNNING ON CPU


class LlamaCppTrainedInferenceSettings:
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    PRESENCE_PENALTY: float = 0.0
    REPEAT_PENALTY: float = 1.1
    TOP_K: int = 20
    TOP_P: float = 0.95


class TrainingSettings:
    pass