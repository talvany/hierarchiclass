from abc import ABC, abstractmethod
import torch
import os
import pickle
from codebase.settings import LOOKUP_PKL_FILENAME


class ModelPredictor(ABC):
    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hierarchy_lookup_dict = load_lookup_dict_from_pkl(LOOKUP_PKL_FILENAME)

    def predict(self, sentences):
        """
        Get hierarchical predictions for sentences
        :param sentences: the senteces
        :return: tuples with the predictiosn
        """

        result = self.get_predictions(sentences)

        result_list = result.label.to_list()

        tps = [self.hierarchy_lookup_dict[pred] for pred in result_list]
        return tps

    @abstractmethod
    def get_predictions(self, sentences):
        pass


def load_lookup_dict_from_pkl(filename):
    """
    Loads the pickle for hierarchy lookup
    :return: the dictionary
    """
    if not os.path.exists(filename):
        raise IOError(f"{filename} file does not exist")

    with open(LOOKUP_PKL_FILENAME, "rb") as infile:
        return pickle.load(infile)
