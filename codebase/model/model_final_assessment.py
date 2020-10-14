"""
Evaluates a dataset (usually the test set) in a model
"""

import click
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from transformers import XLNetForSequenceClassification
import pandas as pd
from codebase.model.model_data_handler import get_inputs, get_dataloader, generate_dataloader_input

from codebase.constants import BATCH_NUM
from codebase.log import logger
from codebase.model.model_data_handler import get_dataloader, get_inputs
from codebase.model.model_trainer import ModelTrainer
from codebase.settings import LOOKUP_PKL_FILENAME
from codebase.util import get_existing_tag2idx


@click.command()
@click.option(
    "--test-set",
    required=True,
    help="Path to the csv file containing the test set ",
    type=click.Path(exists=True),
)
@click.option(
    "--save-folder",
    required=True,
    help="Path where to save the model",
    type=click.Path(),
)
@click.option(
    "--model-name", required=True, help="The name given to the model", type=click.STRING
)
@click.option(
    "--label-column-name", default="l3", help="The name of the column containing the labels", type=click.STRING
)
def main(test_set, save_folder, model_name, label_column_name='labels'):
    # joins the path of the model folder
    model_folder = os.path.join(save_folder, model_name)

    test_df = pd.read_csv(test_set)

    sentences = test_df.text.to_list()
    labels = test_df[label_column_name].to_list()

    tag2idx = get_existing_tag2idx(model_folder)

    logger.info("Setting input embedding...")

    (
        full_input_ids,
        full_input_masks,
        full_segment_ids,
        tags,
    ) = generate_dataloader_input(sentences, labels, tag2idx)

    test_dataloader = get_dataloader(
        full_input_ids, full_input_masks, full_segment_ids, BATCH_NUM, tags
    )

    trainer = ModelTrainer(model_folder)
    trainer.evaluate_model(test_dataloader, output_results_filename="test_eval.txt")


if __name__ == "__main__":
    main()
