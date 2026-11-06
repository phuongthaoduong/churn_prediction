from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
LLM_RAW = DATA / "llm_raw"
PIPELINE_DIR = HERE.parent / "pipeline"
EXPERIMENT_DIR = HERE.parent / "experiment_output"

SEED = 42
DECISION_OPTIONS = ["hold", "retrain", "reset_reference"]
