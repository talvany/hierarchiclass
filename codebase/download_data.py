"""
This module downloads data for data training and for model execution
"""
import click
import wget
import gdown
import tarfile
import os
from codebase.log import logger
from codebase.settings import DATA_PATH, OUT_MODELS_PATH, XLNET_BASE_PATH


# the ids of the gdrive files for different modes
GDRIVE_IDS = {
    "main_dataset": [
        ("1kdFlCq9VWsQBhRBlijPis3a2Vhv220p5", DATA_PATH, "classification_dataset.csv"),
        ("1TDiZ9mn9z9J7dTSx9WCz8BURx6yT7IPk", DATA_PATH, "training_all.csv"),
        ("17DaIICz1c1qkZbrdR9qgipXJ-WCye0ow", DATA_PATH, "test.csv"),
    ],
    "train_multi": [
        ("1MvHYpXICuDSmq4jenK9j2_x54hrO21_k", DATA_PATH, "training_c1.csv"),
        ("1E0lL9ZHwNtwZJsVo6iLirNJ7gtkOgFjr", DATA_PATH, "training_c2.csv"),
        ("18hwo7ParTOvlV9WpPrb38Ox73DcCOnYV", DATA_PATH, "training_c3.csv"),
        ("1ZrvkETgLAaYLyfUl47cWGmtIQpQlhDuD", DATA_PATH, "training_c4.csv"),
        ("1uHvwuTxPDrEUzHHirHKS5U5wCLVOUpQ2", DATA_PATH, "training_c5.csv"),
        ("1BBAUiCQnwjK42Fz2Kaih-DmCoVJX6Dtn", DATA_PATH, "training_c6.csv"),
        ("1dKL9O6W75s3qsVQw56C8T8XeeAKIY4ir", DATA_PATH, "training_c7.csv"),
    ],
    "train_single": [
        ("1vAhqiJXBTHp2sb4fOegJrKhZzM3RIGYc", DATA_PATH, "training_balanced.csv"),
    ],
    "predict_single": [
        ("1IDQ-cuqCIqPrZ15V49L9TnggvBSZXc3_", OUT_MODELS_PATH, "balanced.tar.gz")
    ],
    "predict_multi": [
        ("1-KG97zumrdzkkEKMT24vy5MeN971bN1f", OUT_MODELS_PATH, "multiclass.tar.gz")
    ],
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
        download_gdrive_docs([(doc_id, directory, output_name)])
        file_path = os.path.join(directory, output_name)

        logger.info(f"Extracting {file_path}")
        with tarfile.open(file_path) as my_tar:
            my_tar.extractall(directory)


def download_xlnet_data(full=True):
    """
    Download data for xlnet training
    :param full: whether to download all files or just the vocabulary
    :return:
    """
    path_base = "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased"

    if full:
        filenames = ["spiece.model", "pytorch_model.bin", "config.json"]
    else:
        filenames = ["spiece.model"]

    for name in filenames:
        logger.info(f"Downloading {name} for xlnet")
        out_path = os.path.join(XLNET_BASE_PATH, name)

        if not os.path.isfile(out_path):
            wget.download(f"{path_base}-{name}", out=out_path)


@click.command()
@click.option(
    "--mode",
    default=None,
    help="The mode indicating what to download.",
    type=click.Choice(
        [
            "prediction_models_single",
            "prediction_models_multi",
            "train_data_single",
            "train_data_multi",
        ]
    ),
)
def main(mode):
    logger.info(f"Downloading data for mode {mode}")

    # download the main datasets in all modes
    download_gdrive_docs(GDRIVE_IDS["main_dataset"])

    # Download the XLNet vocabulary
    download_xlnet_data(full=False)

    if mode == "prediction_models_single":
        download_and_extract(GDRIVE_IDS["predict_single"])
    elif mode == "prediction_models_multi":
        download_and_extract(GDRIVE_IDS["predict_multi"])
    elif mode == "train_data_single":
        download_xlnet_data()
        download_gdrive_docs(GDRIVE_IDS["train_single"])
    elif mode == "train_data_multi":
        download_xlnet_data()
        download_gdrive_docs(GDRIVE_IDS["train_multi"])

    logger.info("Finished downloading all the data")


if __name__ == "__main__":
    main()
