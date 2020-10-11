import click
from transformers import logging
import os

from codebase.model.model_predictor import ModelPredictor

logging.set_verbosity_error()


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
    model_folder = os.path.join(save_folder, model_name)

    predictor = ModelPredictor(model_folder)
    print(f"Sentence to predict:\n{sentence}\n")

    predictor.predict(sentence)



if __name__ == "__main__":
    main()
