import pickle
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from codebase.model.inputs import get_inputs
from codebase.tokenizerwrapper import TokenizerWrapper

class ModelPredictor:

    def __init__(self, model_folder):
        self.model_folder = model_folder

    def predict(self, sentence):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()


        tag2idx_file = os.path.join(self.model_folder, "tag2idx.bin")
        self.tag2idx = torch.load(tag2idx_file)



        tag2name = {self.tag2idx[key]: key for key in self.tag2idx.keys()}

        model = XLNetForSequenceClassification.from_pretrained(
            self.model_folder, num_labels=len(tag2name)
        )
        model.to(device)

        model.eval()


        sentences = [sentence]

        print("Setting input embedding")


        max_len = 64

        full_input_ids = []
        full_input_masks = []
        full_segment_ids = []

        self.tokenizer = TokenizerWrapper().tokenizer


        for i, sentence in tqdm(enumerate(sentences), total=len(sentences)):
            input_ids, input_mask, segment_ids = get_inputs(sentence)


            full_input_ids.append(input_ids)
            full_input_masks.append(input_mask)
            full_segment_ids.append(segment_ids)

        inputs = torch.tensor(full_input_ids)
        masks = torch.tensor(full_input_masks)
        segs = torch.tensor(full_segment_ids)

        # Set batch num
        batch_num = 32

        data = TensorDataset(inputs, masks, segs)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_num)

        nb_eval_steps, nb_eval_examples = 0, 0

        y_predict = []
        print("***** Running evaluation *****")
        print("  Num examples ={}".format(len(inputs)))
        print("  Batch size = {}".format(batch_num))
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
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