import pickle
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import XLNetForSequenceClassification
from codebase.model.model_data_handler import get_dataloader, get_inputs
from codebase.constants import BATCH_NUM, TAG2IDX_FILENAME
from codebase.settings import LOOKUP_PKL_FILENAME
from codebase.log import logger

class ModelPredictor:

    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hierarchy_lookup_dict = load_lookup_dict_from_pkl(LOOKUP_PKL_FILENAME)

    def predict(self, sentences):
        tag2idx_file = os.path.join(self.model_folder, TAG2IDX_FILENAME)

        if not os.path.exists(tag2idx_file):
            raise IOError(f"{tag2idx_file} file does not exist")

        self.tag2idx = torch.load(tag2idx_file)

        tag2name = {self.tag2idx[key]: key for key in self.tag2idx.keys()}

        model = XLNetForSequenceClassification.from_pretrained(
            self.model_folder, num_labels=len(tag2name)
        )
        model.to(self.device)
        model.eval()

        logger.info("Setting input embedding")

        input = []
        masks = []
        segs = []

        for i, sentence in tqdm(enumerate(sentences), total=len(sentences)):
            input_ids, input_mask, segment_ids = get_inputs(sentence)

            input.append(input_ids)
            masks.append(input_mask)
            segs.append(segment_ids)

        dataloader = get_dataloader(input, masks, segs, BATCH_NUM)

        nb_eval_steps, nb_eval_examples = 0, 0

        y_predict = []
        logger.info("Running evaluation...")

        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_segs = batch

            with torch.no_grad():
                outputs = model(
                    input_ids=b_input_ids,
                    token_type_ids=b_segs,
                    input_mask=b_input_mask,
                )
                logits = outputs[0]

            # Get text classification predict result
            logits = logits.detach().cpu().numpy()

            for predict in np.argmax(logits, axis=1):
                y_predict.append(predict)

            nb_eval_steps += 1

        tps = [self.hierarchy_lookup_dict[tag2name[pred]] for pred in y_predict]
        return tps


def load_lookup_dict_from_pkl(filename):
    """
    Loads the pickle for hierarchy lookup
    :return: the dictionary
    """
    if not os.path.exists(filename):
        raise IOError(f"{filename} file does not exist")

    with open(LOOKUP_PKL_FILENAME, "rb") as infile:
        return pickle.load(infile)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
