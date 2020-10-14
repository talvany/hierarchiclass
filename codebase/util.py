"""
Utility functions
"""
import os
import numpy as np
from codebase.log import logger
from codebase.constants import TAG2IDX_FILENAME
import torch

def accuracy(out, labels):
    """
    Calculates the accuracy based on output and labels
    :param out: the output from the modoel
    :param labels: the labels
    :return:
    """
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def print_classification(sentences, tps):
    """
    Print the classification
    :param sentences: the sentences that were classifier
    :param tps: tuples with the classifications
    :return:
    """
    for i in range(len(sentences)):
        logger.info(sentences[i])
        tp = tps[i]
        logger.info(f"Level 1: {tp[0]}")
        logger.info(f"Level 2: {tp[1]}")
        logger.info(f"Level 3: {tp[2]}")
        logger.info("")

def get_existing_tag2idx(model_folder):
    """
    Gets an existing tag2idx object from the folder of the model
    :param model_folder:
    :return:
    """
    tag2idx_file = os.path.join(model_folder, TAG2IDX_FILENAME)

    if not os.path.exists(tag2idx_file):
        raise IOError(f"{tag2idx_file} file does not exist")

    tag2idx = torch.load(tag2idx_file)
    return tag2idx