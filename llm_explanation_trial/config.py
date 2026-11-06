from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
LLM_RAW = DATA / "llm_raw"
EXPERIMENT_DIR = HERE.parent / "experiment_output"

SEED = 42
N_TOP_RISK = 10
N_RANDOM = 10
TOP_K_FEATURES = 5
