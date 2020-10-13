# hierarchiclass

This project consists of a hierarchical text classification task. 

1) Exploring the dataset 

To see details about the data analysis carried out on the data and the chosen approach, check the first notebook in the notebooks directory ("1 - Analyse data.ipynb")

2) Model selection

I decided to use [XLNet](https://arxiv.org/pdf/1906.08237.pdf) as my model architecture.

Some of the reasons that made me go with XLNet:

- XLNet is still the state of the art for text classification in many datasets (e.g. the AG News corpus, the DBpedia dataset and 	
IMDb), so I knew it would yield good results.
- The data provided for this project is similar to dbpedia data, which XLNet achieves the state of the art, so I had reason to believe that this would be a good choice.
- On a personal note, I had not explored with XLNet in practice before, so I was curious to try.




 