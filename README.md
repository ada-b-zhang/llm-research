## tokenization.ipynb
- Notebook to explore different tokenizers
- Example dataset from Kaggle provided, but you can upload any text dataset

## flores_plus
Folder that contains files to calculate fertility and parity scores for Meta's [FLORES+ dataset](https://huggingface.co/datasets/openlanguagedata/flores_plus).
- `fertility` folder contains fertilty calculations and visualizations
- `parity` folder contains parity calculations and visualizations

## AfriMMLU
Folder that contains script to calculate fertility scores for [AfriMMLU dataset](https://huggingface.co/datasets/masakhane/afrimmlu).


## MoverScore
Folder with scripts to calculate MoverScore.
1. [Clone the original repository](https://github.com/AIPHES/emnlp19-moverscore.git) ```git clone https://github.com/AIPHES/emnlp19-moverscore.git```
2. Add `moverscore/exploring_moverscore.ipynb` from this repo to the cloned repo
3. Replace `moverscore/moverscore_v2.py` in cloned repo with `moverscore_v2.py` from this repo

Source: 
@inproceedings{zhao2019moverscore,
  title = {MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance},
  month = {August},
  year = {2019},
  author = {Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger},
  address = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
}
