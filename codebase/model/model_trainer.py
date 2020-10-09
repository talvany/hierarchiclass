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

from codebase.constants import SEG_ID_A, SEG_ID_CLS, SEG_ID_PAD
from codebase.util import accuracy
from codebase.settings import XLNET_BASE_PATH, VOCABULARY_PATH
from codebase.model.inputs import get_inputs

from transformers import logging
logging.set_verbosity_error()


class ModelTrainer:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    batch_num = 32

    def create_dataloaders(self, input_csv):
        """
        :param input_csv: csv with two columns named text and labels
        :return:
        """
        print("Loading data from csv")

        df = pd.read_csv(input_csv)

        sentences = df.text.to_list()
        labels = df.labels.to_list()

        self.tag2idx = {t: i for i, t in enumerate(set(labels))}

        self.tokenizer = XLNetTokenizer(vocab_file=str(VOCABULARY_PATH), do_lower_case=False)
        self.CLS_ID = self.tokenizer.encode("<cls>")[0]
        self.SEP_ID = self.tokenizer.encode("<sep>")[0]

        print("Setting input embedding")

        full_input_ids = []
        full_input_masks = []
        full_segment_ids = []

        for _, sentence in tqdm(enumerate(sentences), total=len(sentences)):
            input_ids, input_mask, segment_ids = get_inputs(sentence)

            full_input_ids.append(input_ids)
            full_input_masks.append(input_mask)
            full_segment_ids.append(segment_ids)

        tags = [self.tag2idx[str(lab)] for lab in labels]

        print("Splitting data into tensors")

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

        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)
        tr_segs = torch.tensor(tr_segs)
        val_segs = torch.tensor(val_segs)

        # Set token, attention and segment embeddings
        train_data = TensorDataset(tr_inputs, tr_masks, tr_segs, tr_tags)
        train_sampler = RandomSampler(train_data)

        self.train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.batch_num, drop_last=True
        )

        valid_data = TensorDataset(val_inputs, val_masks, val_segs, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        self.valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=self.batch_num
        )



    def train_model(self, model_folder, save_model=True, epochs=5, max_grad_norm=1.0):
        """
        :param save_folder: where to save the model
        :param model_name: the name the model will receive
        :param save_model:  whether to save the model or not
        :return:
        """
        print("Train model")

        model = XLNetForSequenceClassification.from_pretrained(
            XLNET_BASE_PATH, num_labels=len(self.tag2idx)
        )

        model.to(ModelTrainer.device)
        if ModelTrainer.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Calculate train optimization num
        num_train_optimization_steps = (
            int(math.ceil(len(self.train_dataloader.dataset) / self.batch_num) / 1)
            * epochs
        )

        # Fine tune model all layer parameters

        param_optimizer = list(model.named_parameters())
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

        model.train()

        print("----- Running training -----")
        print("  Num examples = %d" % (len(self.train_dataloader.dataset)))
        print("  Batch size = %d" % (self.batch_num))
        print("  Num steps = %d" % (num_train_optimization_steps))
        for _ in trange(epochs, desc="Epoch"):
            self.tr_loss = 0
            nb_tr_examples, self.nb_tr_steps = 0, 0
            for step, batch in enumerate(self.train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(ModelTrainer.device) for t in batch)
                b_input_ids, b_input_mask, b_segs, b_labels = batch

                # forward pass
                outputs = model(
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
                    parameters=model.parameters(), max_norm=max_grad_norm
                )

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

            # print train loss per epoch
            print("Train loss: {}".format(self.tr_loss / self.nb_tr_steps))

        # Make save folder if it does not exists
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # Save a trained model, configuration and tokenizer
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(model_folder,  "pytorch_model.bin")
        output_config_file = os.path.join(model_folder, "config.json")
        tag2idx_file = os.path.join(model_folder, "tag2idx.bin")

        if save_model:
            print('Saving the model')

            # Save model into file
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            self.tokenizer.save_vocabulary(model_folder)

            # save tag2idx pickle
            torch.save(self.tag2idx, tag2idx_file)

    def evaluate_model(self, model_folder):
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
        print("  Num examples ={}".format(self.valid_dataloader.dataset))
        print("  Batch size = {}".format(self.batch_num))
        for step, batch in enumerate(self.valid_dataloader):
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
        eval_accuracy = eval_accuracy / len(self.valid_dataloader.dataset)
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