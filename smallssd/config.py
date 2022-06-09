from pathlib import Path


DATAFOLDER_PATH = Path(__file__).parent.parent / "data"

TRAINING_DATAFOLDER_NAME = "training_data"
EVAL_DATAFOLDER_NAME = "eval_data"
UNLABELLED_DATAFOLDER_NAME = "unlabelled_data"

TRAIN_DATA_PATH = DATAFOLDER_PATH / TRAINING_DATAFOLDER_NAME
EVAL_DATA_PATH = DATAFOLDER_PATH / EVAL_DATAFOLDER_NAME

CROP_ANNOTATIONS = "crop_annotations"
IMAGES = "images"
WEED_ANNOTATIONS = "weed_annotations"

DATASET_VERSION_ID = 6627697
DATASET_URL = f"https://zenodo.org/record/{DATASET_VERSION_ID}"
