
import numpy as np
import torch
import pandas as pd
from transformers import XLNetForSequenceClassification
from codebase.model.data_handler import get_dataloader, generate_dataloader_input
from codebase.constants import BATCH_NUM
from codebase.model.predictor import ModelPredictor
from codebase.util import get_existing_tag2idx
from codebase.log import logger

class ModelPredictorSingleModel(ModelPredictor):

    def __init__(self, model_folder):
        super().__init__(model_folder)


    def get_predictions(self, sentences):
        """
        Get the string predictions for each sentence
        :param sentences: the sentences
        :return: a dataframe containing the sentences and the predictions
        """
        """
        Makes prediction on sentences
        :param sentences: the sentences
        :return: a dataframe a dataframe with sentences and predictions
        """
        self.tag2idx = get_existing_tag2idx(self.model_folder)
        tag2name = {self.tag2idx[key]: key for key in self.tag2idx.keys()}

        model = XLNetForSequenceClassification.from_pretrained(
            self.model_folder, num_labels=len(tag2name)
        )
        model.to(self.device)
        model.eval()

        logger.info("Setting input embedding")

        input, masks, segs = generate_dataloader_input(sentences)
        dataloader = get_dataloader(input, masks, segs, BATCH_NUM)

        nb_eval_steps, nb_eval_examples = 0, 0

        y_predict = []
        logger.info("Running evaluation...")

        for step, batch in enumerate(dataloader):
            if nb_eval_steps % 100 == 0:
                logger.info(f"Step {nb_eval_steps}")

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


        final_df = pd.DataFrame(
            {
                "sentences": sentences,
                "label": [tag2name[pred] for pred in y_predict],
                "y_pred": y_predict
             }
        )

        return final_df

