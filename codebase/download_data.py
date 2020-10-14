"""
This module downloads data for data training:
- the files necessary to run XLNet using huggingface
- the original csv for the training data: training_balanced.csv
- the new datasets generated based on the original dataset
"""
import click
import wget
import gdown
import tarfile
import os


def download_data(gdrive_ids, download_xlnet=True, multimodel=False):
    if download_xlnet:
        path_base = (
            "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased"
        )

        filenames = ["spiece.model", "pytorch_model.bin", "config.json"]

        for name in filenames:
            print(name)
            wget.download(
                f"{path_base}-{name}", out=f"../models/xlnet-base-cased/{name}"
            )
    if multimodel:
        doc_id = '1-hvWGEeRECSK8WQPwfzjlSePf3MUjFnI'
        directory = "../models/out/"
        file_name = "multiclass.tar.gz"
        file_path = os.path.join(directory, file_name)

        url = f"https://drive.google.com/uc?id={doc_id}"
        gdown.download(url, file_path, quiet=False)

        with tarfile.open(file_path) as my_tar:
            my_tar.extractall(directory)

    for doc_id, output in gdrive_ids:
        url = f"https://drive.google.com/uc?id={doc_id}"
        gdown.download(url, output, quiet=False)


@click.command()
def main():
    gdrive_ids = [
        # ("1kdFlCq9VWsQBhRBlijPis3a2Vhv220p5", "../data/classification_dataset.csv"),
        # ("1TDiZ9mn9z9J7dTSx9WCz8BURx6yT7IPk", "../data/training_all.csv"),
        # ("17DaIICz1c1qkZbrdR9qgipXJ-WCye0ow", "../data/test.csv"),
        # ("1MvHYpXICuDSmq4jenK9j2_x54hrO21_k", "../data/training_c1.csv"),
        # ("1E0lL9ZHwNtwZJsVo6iLirNJ7gtkOgFjr", "../data/training_c2.csv"),
        # ("18hwo7ParTOvlV9WpPrb38Ox73DcCOnYV", "../data/training_c3.csv"),
        # ("1ZrvkETgLAaYLyfUl47cWGmtIQpQlhDuD", "../data/training_c4.csv"),
        ("1uHvwuTxPDrEUzHHirHKS5U5wCLVOUpQ2", "../data/training_c5.csv"),
        # ("1BBAUiCQnwjK42Fz2Kaih-DmCoVJX6Dtn", "../data/training_c6.csv"),
        # ("1dKL9O6W75s3qsVQw56C8T8XeeAKIY4ir", "../data/training_c7.csv"),
        # ("1vAhqiJXBTHp2sb4fOegJrKhZzM3RIGYc", "../data/training_balanced.csv"),
    ]
    download_data(gdrive_ids, download_xlnet=True, multimodel=False)
    print("Finished downloading data")


if __name__ == "__main__":
    main()
