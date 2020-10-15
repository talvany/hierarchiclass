"""
Executes a prediction given a model name
"""
import click
import transformers
import os
from codebase.log import logger
from codebase.util import print_classification
from codebase.model.predictor_single_model import ModelPredictorSingleModel
from codebase.model.predictor_multimodel import ModelPredictorMultiModel


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
    "--model-name", default=None, help="The name given to the model", type=click.STRING
)
@click.option('--multimodel', default=False,
              help="Whether to use the multimodel approach", type=click.BOOL)
def main(sentence, save_folder, model_name, multimodel):

    if multimodel:
        predictor = ModelPredictorMultiModel(save_folder)
    else:
        # if it's the single model approach
        model_folder = os.path.join(save_folder, model_name)
        predictor = ModelPredictorSingleModel(model_folder)

    tps = predictor.predict([sentence])

    print_classification([sentence], tps)


if __name__ == "__main__":
    main()
