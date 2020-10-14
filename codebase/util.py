"""
Utility functions
"""
import numpy as np
from codebase.log import logger


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
