"""
Settings for the application
"""
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(PACKAGE_ROOT)

DATA_PATH = PACKAGE_ROOT.joinpath("data")
MODEL_PATH = PACKAGE_ROOT.joinpath("models")
MISC_PATH = PACKAGE_ROOT.joinpath("misc")

XLNET_BASE_PATH = MODEL_PATH.joinpath("xlnet-base-cased")
OUT_MODELS_PATH = MODEL_PATH.joinpath("out")
VOCABULARY_PATH = XLNET_BASE_PATH.joinpath("spiece.model")

LOOKUP_PKL_FILENAME = MISC_PATH.joinpath("hierarchy_lookup_dict.pkl")
