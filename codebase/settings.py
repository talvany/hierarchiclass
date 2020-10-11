import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PACKAGE_ROOT.joinpath('data')
MODEL_PATH = PACKAGE_ROOT.joinpath('models')

XLNET_BASE_PATH = MODEL_PATH.joinpath('xlnet-base-cased')

VOCABULARY_PATH =  XLNET_BASE_PATH.joinpath('spiece.model')

