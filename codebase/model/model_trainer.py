import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from tqdm import tqdm, trange
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from codebase.constants import (
    TAG2IDX_FILENAME,
    CONFIG_FILENAME,
    PYTORCH_MODEL_NAME,
    BATCH_NUM,
)
from codebase.model.model_data_handler import get_inputs, get_dataloader, generate_dataloader_input
from codebase.settings import XLNET_BASE_PATH
from codebase.util import accuracy, get_existing_tag2idx
from codebase.log import logger


class ModelTrainer:

    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.tag2idx = None
        self.tr_loss = None

    def split_into_dataloaders(self, input_csv):
        """
        :param input_csv: csv with two columns named text and labels
        :return:
        """
        logger.info("Loading data from csv...")

        df = pd.read_csv(input_csv)

        sentences = df.text.to_list()
        labels = df.labels.to_list()

        self.tag2idx = {t: i for i, t in enumerate(set(labels))}

        logger.info("Setting input embedding...")

        full_input_ids, full_input_masks, full_segment_ids, tags = generate_dataloader_input(sentences, labels, self.tag2idx)

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

        logger.info("Getting dataloaders...")

        train_dataloader = get_dataloader(
            tr_inputs, tr_masks, tr_segs, BATCH_NUM, tr_tags
        )
        valid_dataloader = get_dataloader(val_inputs, val_masks, val_segs, BATCH_NUM, val_tags)

        return train_dataloader, valid_dataloader

    def train_model(self, train_dataloader, epochs=5, max_grad_norm=1.0):
        """
        :param save_folder: where to save the model
        :param model_name: the name the model will receive
        :param save_model:  whether to save the model or not
        :return:
        """
        logger.info("Preparing for training...")

        self.model = XLNetForSequenceClassification.from_pretrained(
            XLNET_BASE_PATH, num_labels=len(self.tag2idx)
        )

        self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Calculate train optimization num
        num_train_optimization_steps = (
            int(math.ceil(len(train_dataloader.dataset) / BATCH_NUM) / 1) * epochs
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

        logger.info("----- Running training -----")
        logger.info("  Num examples = %d" % (len(train_dataloader.dataset)))
        logger.info("  Batch size = %d" % (BATCH_NUM))
        logger.info("  Num steps = %d" % (num_train_optimization_steps))
        for i in trange(epochs, desc="Epoch"):
            self.tr_loss = 0
            self.nb_tr_steps = 0
            for step, batch in enumerate(train_dataloader):
                if self.nb_tr_steps % 100 == 0:
                    logger.info(f"Step {self.nb_tr_steps}")
                # add batch to gpu
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_segs, b_labels = batch

                # forward pass
                outputs = self.model(
                    input_ids=b_input_ids,
                    token_type_ids=b_segs,
                    input_mask=b_input_mask,
                    labels=b_labels,
                )
                loss, logits = outputs[:2]
                if self.n_gpu > 1:
                    # When multi gpu, average it
                    loss = loss.mean()

                # backward pass
                loss.backward()

                # track train loss
                self.tr_loss += loss.item()
                self.nb_tr_steps += 1

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=max_grad_norm
                )

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

            # print train loss per epoch
            logger.info("Epoch: {}".format(i))
            logger.info("Train loss: {}".format(self.tr_loss / self.nb_tr_steps))

    def save_model(self):
        if not self.model:
            raise IOError(f"{tag2idx_file} No model to save")

        # Make save folder if it does not exists
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        # Save a trained model and configuration
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        output_model_file = os.path.join(self.model_folder, PYTORCH_MODEL_NAME)
        output_config_file = os.path.join(self.model_folder, CONFIG_FILENAME)
        tag2idx_file = os.path.join(self.model_folder, TAG2IDX_FILENAME)

        logger.info("Saving the model...")

        # Save model into file
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

        # save tag2idx pickle
        torch.save(self.tag2idx, tag2idx_file)

    def evaluate_model(self, dataloader, output_results_filename="eval_results.txt"):
        """
        Evaluates the model
        :param model_folder: the folder of the model
        :param model_name: the name of the model
        :return:
        """

        eval_general_metrics, class_report = self.get_reports(dataloader)
        results_string = self.get_printed_eval_results(eval_general_metrics, class_report)

        # Save the file report
        output_eval_file = os.path.join(self.model_folder, output_results_filename)
        logger.info(results_string)

        with open(output_eval_file, "w") as writer:
            writer.write(results_string)


    def get_reports(self, dataloader):
        logger.info("Loading model for evaluation...")

        if not self.tag2idx:
            self.tag2idx = get_existing_tag2idx(self.model_folder)

        model = XLNetForSequenceClassification.from_pretrained(
            self.model_folder, num_labels=len(self.tag2idx)
        )
        model.to(self.device)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        y_true = []
        y_predict = []

        logger.info("----- Running evaluation -----")
        logger.info("  Num examples ={}".format(len(dataloader.dataset)))
        logger.info("  Batch size = {}".format(BATCH_NUM))
        for step, batch in enumerate(dataloader):
            if self.nb_tr_steps % 100 == 0:
                logger.info(f"Step {self.nb_tr_steps}")

            batch = tuple(t.to(self.device) for t in batch)
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
        eval_accuracy = eval_accuracy / len(dataloader.dataset)


        eval_general_metrics = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}

        # if the object has the training model, we can extract th training loss
        if self.tr_loss:
            loss = self.tr_loss / self.nb_tr_steps
            eval_general_metrics["loss"] = loss

        class_report = classification_report(
            y_pred=np.array(y_predict), y_true=np.array(y_true)
        )

        return eval_general_metrics, class_report


    def get_printed_eval_results(self, general_metrics, report):
        """
        Get the result of the evaluation in a string to be printed
        :param general_metrics: the general metrics
        :param report: the classification report
        :return:
        """
        printed_eval_results = "----- Evaluation results -----"
        for key in sorted(general_metrics.keys()):
            printed_eval_results += "\n  %s = %s" % (key, str(general_metrics[key]))
        printed_eval_results += f"\n{report}"
        return printed_eval_results
