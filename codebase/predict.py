"""
Executes a prediction given a model name
"""
import click
import transformers
import os
from codebase.log import logger
from codebase.util import print_classification
from codebase.model.model_predictor import ModelPredictor
from codebase.model.model_predictor_multimodel import ModelPredictorMultiModel


@click.command()
@click.option(
    "--sentence", required=True, help="The sentence to be classifier", type=click.STRING
)
@click.option(
    "--save-folder",
    required=True,
    help="Path where to save the model",
    type=click.Path(),
)
@click.option(
    "--model-name", required=True, help="The name given to the model", type=click.STRING
)
def main(sentence, save_folder, model_name):
    # joins the path of the model folder
    model_folder = os.path.join(save_folder, model_name)

    # predictor = ModelPredictor(model_folder)
    predictor = ModelPredictorMultiModel(save_folder) # TODO FIX

    logger.info(f"Sentence to predict:")
    logger.info(f"{sentence}")

    tps = predictor.predict([sentence])
    print(tps)
    # print_classification([sentence], tps)


if __name__ == "__main__":
    main()
