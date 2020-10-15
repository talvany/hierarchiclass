"""
Evaluates a dataset (usually the test set) in a model
"""

import click
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report
from codebase.model.predictor_multimodel import ModelPredictorMultiModel
from codebase.model.predictor_single_model import ModelPredictorSingleModel


@click.command()
@click.option(
    "--test-set",
    required=True,
    help="Path to the csv file containing the test set ",
    type=click.Path(exists=True),
)
@click.option(
    "--save-folder",
    required=True,
    help="Path where to save the model",
    type=click.Path(),
)
@click.option(
    "--model-name", default=None, help="The name given to the model", type=click.STRING
)
@click.option(
    "--label-column-name",
    default="l3",
    help="The name of the column containing the labels",
    type=click.STRING,
)
@click.option(
    "--multimodel",
    default=False,
    help="Whether to use the multimodel approach",
    type=click.BOOL,
)
def main(test_set, save_folder, model_name, label_column_name, multimodel):
    # joins the path of the model folder

    test_df = pd.read_csv(test_set)

    sentences = test_df.text.to_list()
    labels = test_df[label_column_name].to_list()

    if multimodel:
        model_folder = save_folder
        predictor = ModelPredictorMultiModel(save_folder)
    else:
        model_folder = os.path.join(save_folder, model_name)
        predictor = ModelPredictorSingleModel(model_folder)

    predictions_df = predictor.get_predictions(sentences)

    class_report = classification_report(
        y_pred=np.array(predictions_df.label.to_list()), y_true=np.array(labels)
    )
    print(class_report)

    with open(os.path.join(model_folder, "final_assessment.txt"), "w") as writer:
        writer.write(class_report)


if __name__ == "__main__":
    main()
