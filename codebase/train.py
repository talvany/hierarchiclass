import click
import os
from codebase.model.model_trainer import ModelTrainer


@click.command()
@click.option(
    "--input-csv",
    required=True,
    help="Path to the csv file containing input data ",
    type=click.Path(exists=True),
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
def main(input_csv, save_folder, model_name):
    trainer = ModelTrainer()

    model_folder = os.path.join(save_folder, model_name)

    train_dataloader, valid_dataloader = trainer.split_into_dataloaders(input_csv)
    trainer.train_model(model_folder, train_dataloader, epochs=1)
    trainer.evaluate_model(model_folder, valid_dataloader)
    trainer.save_model()


if __name__ == "__main__":
    main()