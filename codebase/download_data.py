"""
This module downloads data for data training and for model execution
"""
import click
import wget
import gdown
import tarfile
import os
from codebase.log import logger
from codebase.settings import DATA_PATH, OUT_MODELS_PATH


# the ids of the gdrive files for different modes
GDRIVE_IDS = {
    'main_dataset' : [
        ("1kdFlCq9VWsQBhRBlijPis3a2Vhv220p5", DATA_PATH, "classification_dataset.csv"),
        ("1TDiZ9mn9z9J7dTSx9WCz8BURx6yT7IPk", DATA_PATH, "training_all.csv"),
        ("17DaIICz1c1qkZbrdR9qgipXJ-WCye0ow", DATA_PATH, "test.csv"),
    ],
    'train_multi' : [
        ("1MvHYpXICuDSmq4jenK9j2_x54hrO21_k", DATA_PATH, "training_c1.csv"),
        ("1E0lL9ZHwNtwZJsVo6iLirNJ7gtkOgFjr", DATA_PATH, "training_c2.csv"),
        ("18hwo7ParTOvlV9WpPrb38Ox73DcCOnYV", DATA_PATH, "training_c3.csv"),
        ("1ZrvkETgLAaYLyfUl47cWGmtIQpQlhDuD", DATA_PATH, "training_c4.csv"),
        ("1uHvwuTxPDrEUzHHirHKS5U5wCLVOUpQ2", DATA_PATH, "training_c5.csv"),
        ("1BBAUiCQnwjK42Fz2Kaih-DmCoVJX6Dtn", DATA_PATH, "training_c6.csv"),
        ("1dKL9O6W75s3qsVQw56C8T8XeeAKIY4ir", DATA_PATH, "training_c7.csv")
    ],
    'train_single' : [
        ("1vAhqiJXBTHp2sb4fOegJrKhZzM3RIGYc", DATA_PATH, "training_balanced.csv"),
    ],
    'predict_single': [
        ("1IDQ-cuqCIqPrZ15V49L9TnggvBSZXc3_", OUT_MODELS_PATH, "balanced.tar.gz")
    ],
    'predict_multi': [
        ("1-KG97zumrdzkkEKMT24vy5MeN971bN1f", OUT_MODELS_PATH, "multiclass.tar.gz")
    ]
}

def download_gdrive_docs(gdrive_ids):
    """
    Downloads a list of documents from google drive
    :param gdrive_ids: list of tuples (<gdrive id>, <destination dir>, <output file name>)
    :return:
    """
    for doc_id, directory, output_name in gdrive_ids:
        # only downloads if file does not exist
        output_filepath = os.path.join(directory, output_name)
        if not os.path.isfile(output_filepath):
            logger.info(f"Downloading {output_filepath}")
            url = f"https://drive.google.com/uc?id={doc_id}"
            gdown.download(url, output_filepath, quiet=False)



def download_and_extract(gdrive_ids):
    """
    Downloads from gdrive and extracts
    :param gdrive_ids: list of tuples (<gdrive id>, <destination dir>, <output file name>)
    :return:
    """
    for doc_id, directory, output_name in gdrive_ids:
        file_path = os.path.join(directory, output_name)

        download_gdrive_docs([(doc_id, file_path, output_name)])

        logger.info(f"Extracting {file_path}")
        with tarfile.open(file_path) as my_tar:
            my_tar.extractall(directory)


def download_xlnet_data():
    """
    Download data for xlnet training
    :return:
    """
    path_base = (
        "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased"
    )

    filenames = ["spiece.model", "pytorch_model.bin", "config.json"]

    for name in filenames:
        logger.info(f'Downloading {name} for xlnet')
        wget.download(
            f"{path_base}-{name}", out=f"../models/xlnet-base-cased/{name}"
        )

@click.command()
@click.option(
    "--mode",
    required=True,
    help="The mode indicating what to download.",
    type=click.Choice(
        ["prediction_models_single", "prediction_models_multi", "train_data_single", "train_data_multi"]
    ),
)
def main(mode):
    logger.info(f'Downloading data for mode {mode}')

    # download the main datasets in all modes
    download_gdrive_docs(GDRIVE_IDS['main_dataset'])

    if mode == "prediction_models_single":
        download_and_extract(GDRIVE_IDS['predict_single'])
    elif mode == "prediction_models_multi":
        download_and_extract(GDRIVE_IDS['predict_multi'])
    elif mode == "train_data_single":
        download_xlnet_data()
        download_gdrive_docs(GDRIVE_IDS['train_data_single_model'])
    elif mode == "train_data_multi":
        download_xlnet_data()
        download_gdrive_docs(GDRIVE_IDS['train_data_multi_model'])

    logger.info("Finished downloading all the data")


if __name__ == "__main__":
    main()
