# hierarchiclass

## Overview

This project consists of a hierarchical text classification task. 

## Usage

First, make sure your requirements are installed and your env variables set:
```
cd hierarchiclass
export PYTHONPATH="."
pip install -r requirements.txt
```

To predict a model using the single model approach:
```
python codebase/predict.py --sentence <SENTENCE> --save-folder models/out/ --model-name balanced
```


To predict a model using the multi model approach:
```
python codebase/predict.py --sentence <SENTENCE> --save-folder models/out/multimodel --multimodel True
```

To train a model:
```
python codebase/train.py --input-csv <INPUT_CSV> --save-folder <SAVE_FOLDER> --model-name <MODEL_NAME>
```

To run evaluation for the single model approach on the test set:

python codebase/model/final_assessment.py --test-set data/test.csv --save-folder models/out/ --model-name balancedl3


python codebase/model/final_assessment.py --test-set data/test.csv --save-folder models/out/multimodel  --multimodel True

## Report

1) Exploring the dataset 

To see details about the data analysis carried out on the data and the chosen approach, check the first notebook in the notebooks directory ("1 - Analyse data.ipynb")

2) Model selection

I decided to use [XLNet](https://arxiv.org/pdf/1906.08237.pdf) as my model architecture.

Some of the reasons that made me go with XLNet:

- XLNet is still the state of the art for text classification in many datasets (e.g. the AG News corpus, the DBpedia dataset and 	
IMDb), so I knew it would yield good results.
- The data provided for this project is similar to dbpedia data, which XLNet achieves the state of the art, so I had reason to believe that this would be a good choice.
- On a personal note, I had not explored with XLNet in practice before, so I was curious to try.

3) Results

I tried a single model approach and a multi model approach to solve this problem. My initial hypothesis was that 