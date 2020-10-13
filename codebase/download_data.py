import click

import wget
import gdown

def download_data(gdrive_ids, download_xlnet=True):
    if download_xlnet:
        path_base = 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased'

        filenames = ['spiece.model', 'pytorch_model.bin', 'config.json']

        for name in filenames:
            print(name)
            wget.download(f'{path_base}-{name}', out=f'../models/xlnet-base-cased/{name}')

    for doc_id, output in gdrive_ids:
        url = f'https://drive.google.com/uc?id={doc_id}'
        gdown.download(url, output, quiet=False)

@click.command()
def main():
    gdrive_ids = [
        # ('1kdFlCq9VWsQBhRBlijPis3a2Vhv220p5','../data/classification_dataset.csv'),
        # ('1vAhqiJXBTHp2sb4fOegJrKhZzM3RIGYc','../data/training_balanced.csv'),
        ('1E0lL9ZHwNtwZJsVo6iLirNJ7gtkOgFjr', '../temp/c2.csv')
    ]
    download_data(gdrive_ids, False)
    print('Finished downloading data')

if __name__ == "__main__":
    main()