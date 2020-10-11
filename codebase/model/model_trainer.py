import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import logging

from codebase.model.model_data_handler import get_inputs, get_dataloader
from codebase.settings import XLNET_BASE_PATH, VOCABULARY_PATH
from codebase.util import accuracy

logging.set_verbosity_error()


class ModelTrainer:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    batch_num = 32
    TAG2IDX_FILENAME = "tag2idx.bin"
    CONFIG_FILENAME = "config.json"
    PYTORCH_MODEL_NAME = "pytorch_model.bin"

    def split_into_dataloaders(self, input_csv):
        """
        :param input_csv: csv with two columns named text and labels
        :return:
        """
        print("Loading data from csv...")

        df = pd.read_csv(input_csv)

        sentences = df.text.to_list()
        labels = df.labels.to_list()

        self.tag2idx = {t: i for i, t in enumerate(set(labels))}

        print("Setting input embedding...")

        full_input_ids = []
        full_input_masks = []
        full_segment_ids = []

        for _, sentence in tqdm(enumerate(sentences), total=len(sentences)):
            input_ids, input_mask, segment_ids = get_inputs(sentence)

            full_input_ids.append(input_ids)
            full_input_masks.append(input_mask)
            full_segment_ids.append(segment_ids)

        tags = [self.tag2idx[str(lab)] for lab in labels]

        # split the data
        (
            tr_inputs,
            val_inputs,
            tr_tags,
            val_tags,
            tr_masks,
            val_masks,
            tr_segs,
            val_segs,
        ) = train_test_split(
            full_input_ids,
            tags,
            full_input_masks,
            full_segment_ids,
            random_state=4,
            test_size=0.3,
        )

        print("Getting dataloaders...")

        train_dataloader = get_dataloader(tr_inputs, tr_masks, tr_segs, batch_num, ßtr_tags)
        valid_dataloader = get_dataloader(val_inputs, val_masks, val_segs, val_tags)

        return train_dataloader, valid_dataloader

    def train_model(self, model_folder, train_dataloader, epochs=5, max_grad_norm=1.0):
        """
        :param save_folder: where to save the model
        :param model_name: the name the model will receive
        :param save_model:  whether to save the model or not
        :return:
        """
        print("Preparing for training...")

        self.model = XLNetForSequenceClassification.from_pretrained(
            XLNET_BASE_PATH, num_labels=len(self.tag2idx)
        )

        self.model.to(ModelTrainer.device)
        if ModelTrainer.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Calculate train optimization num
        num_train_optimization_steps = (
            int(math.ceil(len(train_dataloader.dataset) / self.batch_num) / 1)
            * epochs
        )

        # Fine tune model all layer parameters

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

        optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

        self.model.train()

        print("----- Running training -----")
        print("  Num examples = %d" % (len(train_dataloader.dataset)))
        print("  Batch size = %d" % (self.batch_num))
        print("  Num steps = %d" % (num_train_optimization_steps))
        for _ in trange(epochs, desc="Epoch"):
            self.tr_loss = 0
            nb_tr_examples, self.nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(ModelTrainer.device) for t in batch)
                b_input_ids, b_input_mask, b_segs, b_labels = batch

                # forward pass
                outputs = self.model(
                    input_ids=b_input_ids,
                    token_type_ids=b_segs,
                    input_mask=b_input_mask,
                    labels=b_labels,
                )
                loss, logits = outputs[:2]
                if ModelTrainer.n_gpu > 1:
                    # When multi gpu, average it
                    loss = loss.mean()

                # backward pass
                loss.backward()

                # track train loss
                self.tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                self.nb_tr_steps += 1

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=max_grad_norm
                )

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

            # print train loss per epoch
            print("Train loss: {}".format(self.tr_loss / self.nb_tr_steps))

    def save_model(self, model_folder):
        if not self.model:
            print('No model to save.')
            return

        # Make save folder if it does not exists
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # Save a trained model, configuration and tokenizer
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        output_model_file = os.path.join(model_folder,  ModelTrainer.PYTORCH_MODEL_NAME)
        output_config_file = os.path.join(model_folder, ModelTrainer.CONFIG_FILENAME)
        tag2idx_file = os.path.join(model_folder, ModelTrainer.TAG2IDX_FILENAME)

        print('Saving the model')

        # Save model into file
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)


        # save tag2idx pickle
        torch.save(self.tag2idx, tag2idx_file)

    def evaluate_model(self, model_folder, valid_dataloader):
        """

        :param model_folder: the folder of the model
        :param model_name: the name of the model
        :return:
        """
        print('Loading model')

        if not self.tag2idx:
            tag2idx_file = os.path.join(model_folder, "tag2idx.json")
            self.tag2idx = torch.load(tag2idx_file)

        model = XLNetForSequenceClassification.from_pretrained(
            model_folder, num_labels=len(self.tag2idx)
        )
        model.to(ModelTrainer.device)
        if ModelTrainer.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        y_true = []
        y_predict = []

        print("----- Running evaluation -----")
        print("  Num examples ={}".format(valid_dataloader.dataset))
        print("  Batch size = {}".format(self.batch_num))
        for step, batch in enumerate(valid_dataloader):
            batch = tuple(t.to(ModelTrainer.device) for t in batch)
            b_input_ids, b_input_mask, b_segs, b_labels = batch

            with torch.no_grad():
                outputs = model(
                    input_ids=b_input_ids,
                    token_type_ids=b_segs,
                    input_mask=b_input_mask,
                    labels=b_labels,
                )
                tmp_eval_loss, logits = outputs[:2]

            # Get predictions
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            # Save predictions and gold labels
            for predict in np.argmax(logits, axis=1):
                y_predict.append(predict)

            for real_result in label_ids.tolist():
                y_true.append(real_result)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / len(valid_dataloader.dataset)
        loss = self.tr_loss / self.nb_tr_steps
        result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy, "loss": loss}
        report = classification_report(
            y_pred=np.array(y_predict), y_true=np.array(y_true)
        )

        # Save the file report
        output_eval_file = os.path.join(model_folder, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            print("----- Evaluation results -----")
            for key in sorted(result.keys()):
                print("  %s = %s" % (key, str(result[key])))
                writer.write("%s = %s\n" % (key, str(result[key])))

            print(report)
            writer.write("\n\n")
            writer.write(report)