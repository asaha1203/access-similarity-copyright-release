from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
RUNS_ROOT = PROJECT_ROOT / "runs"

DEFAULT_SPLIT_SEED = 1337
AUTOMATH_DEFAULT_CONFIG = "arxiv-0.50-to-1.00"

TEXT_DATASETS = {
    "automathtext": {
        "hf_path": "math-ai/AutoMathText",
        "hf_name": AUTOMATH_DEFAULT_CONFIG,
        "source_split": "train",
        "materialization": "streaming_rows",
        "materialize_n": 6000,
        "train_full_n": 1000,
        "shard1_n": 500,
        "shard2_n": 500,
        "eval_reserve_n": 5000,
        "text_builder": "title_abstract",
    },
    "writingprompts": {
        "hf_path": "euclaise/writingprompts",
        "hf_name": None,
        "source_split": "merged_train_test",
        "materialization": "merged_train_test",
        "materialize_n": 6000,
        "train_full_n": 1000,
        "shard1_n": 500,
        "shard2_n": 500,
        "eval_reserve_n": 5000,
        "text_builder": "prompt_story",
    },
}

IMAGE_DATASETS = {
    "pokemon": {
        "hf_path": "diffusers/pokemon-gpt4-captions",
        "hf_name": None,
        "source_split": "train",
        "materialization": "hf_rows",
        "materialize_n": 833,
        "train_full_n": 600,
        "shard1_n": 300,
        "shard2_n": 300,
        "eval_reserve_n": 233,
    },
    "laion_mi": {
        "hf_path": "antoniaaa/laion_mi",
        "hf_name": None,
        "source_split": "nonmembers",
        "materialization": "streaming_url_images",
        "materialize_n": 6000,
        "train_full_n": 1000,
        "shard1_n": 500,
        "shard2_n": 500,
        "eval_reserve_n": 5000,
    },
}

