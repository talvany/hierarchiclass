import pickle
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import XLNetForSequenceClassification
from codebase.tokenizerwrapper import TokenizerWrapper
from codebase.model.model_data_handler import get_dataloader, get_inputs

class ModelPredictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_num = 32

    def __init__(self, model_folder):
        self.model_folder = model_folder

    def predict(self, sentence):
        TAG2IDX_FILENAME = "tag2idx.bin"
        tag2idx_file = os.path.join(self.model_folder, TAG2IDX_FILENAME)

        if not os.path.exists(tag2idx_file):
            print(f'{tag2idx_file} file does not exist')
        self.tag2idx = torch.load(tag2idx_file)

        tag2name = {self.tag2idx[key]: key for key in self.tag2idx.keys()}

        model = XLNetForSequenceClassification.from_pretrained(
            self.model_folder, num_labels=len(tag2name)
        )
        model.to(ModelPredictor.device)
        model.eval()

        sentences = [sentence]

        print("Setting input embedding")

        input = []
        masks = []
        segs = []

        self.tokenizer = TokenizerWrapper(self.model_folder).tokenizer


        for i, sentence in tqdm(enumerate(sentences), total=len(sentences)):
            input_ids, input_mask, segment_ids = get_inputs(sentence, self.model_folder)

            input.append(input_ids)
            masks.append(input_mask)
            segs.append(segment_ids)

        dataloader = get_dataloader(input, masks, segs, ModelPredictor.batch_num)

        nb_eval_steps, nb_eval_examples = 0, 0

        y_predict = []
        print("***** Running evaluation *****")
        print("  Num examples ={}".format(len(input)))
        print("  Batch size = {}".format(ModelPredictor.batch_num))

        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(ModelPredictor.device) for t in batch)
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

        print_classification(y_predict, tag2name)

def print_classification(y_predict, tag2name):
    for pred in y_predict:
        tp = lookup_hierarchy(tag2name[pred])
        print(f'Level 1: {tp[0]}')
        print(f'Level 2: {tp[1]}')
        print(f'Level 3: {tp[2]}')


def lookup_hierarchy(l3_label):
    filename = 'misc/hierarchy_lookup_dict.pkl'

    with open(filename, 'rb') as infile:
        new_dict = pickle.load(infile)
    return new_dict[l3_label]


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)