import pickle
import os
import numpy as np
import torch
from transformers import XLNetForSequenceClassification
from codebase.model.data_handler import get_dataloader, generate_dataloader_input
from codebase.model.predictor import ModelPredictor
from codebase.constants import BATCH_NUM
from codebase.log import logger
from codebase.util import get_existing_tag2idx
import pandas as pd


class ModelPredictorMultiModel(ModelPredictor):

    def __init__(self, model_folder):
        super().__init__(model_folder)

    def get_predictions(self, sentences):
        """
        Get the string predictions for each sentence
        :param sentences: the sentences
        :return: a dataframe containing the sentences and the predictions
        """

        c1_folder = os.path.join(self.model_folder, "c1")
        c2_folder = os.path.join(self.model_folder, "c2")
        c3_folder = os.path.join(self.model_folder, "c3")
        c4_folder = os.path.join(self.model_folder, "c4")
        c5_folder = os.path.join(self.model_folder, "c5")
        c6_folder = os.path.join(self.model_folder, "c6")
        c7_folder = os.path.join(self.model_folder, "c7")

        df_c1 = self.predict_in_single_model(sentences, c1_folder)
        df_agent = df_c1[df_c1["label"] == "Agent"]
        df_place = df_c1[df_c1["label"] == "Place"]
        df_other_l1 = df_c1[df_c1["label"] == "Other"]

        df_c2 = self.predict_in_single_model(df_agent["sentences"].to_list(), c2_folder)
        df_athlete = df_c2[df_c2["label"] == "Athlete"]
        df_person = df_c2[df_c2["label"] == "Person"]
        df_other_agent = df_c2[df_c2["label"] == "Other"]

        y_predict_c3 = self.predict_in_single_model(
            df_athlete["sentences"].to_list(), c3_folder
        )
        y_predict_c4 = self.predict_in_single_model(
            df_person["sentences"].to_list(), c4_folder
        )
        y_predict_c5 = self.predict_in_single_model(
            df_other_agent["sentences"].to_list(), c5_folder
        )
        y_predict_c6 = self.predict_in_single_model(
            df_place["sentences"].to_list(), c6_folder
        )
        y_predict_c7 = self.predict_in_single_model(
            df_other_l1["sentences"].to_list(), c7_folder
        )

        l3_pred_df = pd.concat(
            [y_predict_c3, y_predict_c4, y_predict_c5, y_predict_c6, y_predict_c7],
            ignore_index=True,
            sort=False,
        )

        final_df = pd.DataFrame({"sentences": sentences})

        result = pd.merge(final_df, l3_pred_df, on="sentences")

        return result


    def predict_in_single_model(self, sentences, model_folder):
        """
        Executes a prediction in one of the xlnet models
        :param sentences: the sentences
        :param model_folder: the model folder
        :return: a dataframe with sentences and predictions
        """
        if not sentences:
            return pd.DataFrame({"sentences": [], "label": []})

        tag2idx = get_existing_tag2idx(model_folder)
        tag2name = {tag2idx[key]: key for key in tag2idx.keys()}

        model = XLNetForSequenceClassification.from_pretrained(
            model_folder, num_labels=len(tag2name)
        )
        model.to(self.device)
        model.eval()

        logger.info("Setting input embedding for {model_folder}...")

        input, masks, segs = generate_dataloader_input(sentences)
        dataloader = get_dataloader(input, masks, segs, BATCH_NUM)

        nb_eval_steps, nb_eval_examples = 0, 0

        y_predict = []
        logger.info(f"Running evaluation for {model_folder}...")

        for step, batch in enumerate(dataloader):
            if self.nb_tr_steps % 100 == 0:
                logger.info(f"Step {self.nb_tr_steps}")

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

        return pd.DataFrame(
            {"sentences": sentences,
             "label": [tag2name[pred] for pred in y_predict],
             "y_pred": y_predict
             }
        )

