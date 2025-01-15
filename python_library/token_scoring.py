from transformers import AutoTokenizer
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

""" 
This is a library to calculate fertility and parity scores, as well as provide visualizations. 
The default model is `meta-llama/Llama-3.2-1B-Instruct`. 

Functions
---------
    - `token_score`
    - `get_fertilities`
    - `get_parities`

"""

def token_score(text, 
                tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')):
    """ 
    Get the fertility/parity score and tokens for a given text. This is a helper function
    for `get_fertilities` and `get_parities`

    Parameters
    ----------
        - text (str): text for tokenization
        - tokenizer (tokenizer): model/tokenizer 
          (optional, defaults to `meta-llama/Llama-3.2-1B-Instruct`)

    Returns
    -------
        - parity (float): parity score
        - tokenized (list): list of tokens 
    """ 
    tokens = tokenizer.tokenize(text) 
    num_words = len(text.split())

    score = len(tokens) / num_words if num_words > 0 else 999999 
    return score, tokens

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# Fertility 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def get_fertilities(data, 
                    tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct'), 
                    visualize=False):
    """ 
    Get fertility scores and tokens for a dataset of texts in different languages. 
    
    Parameters
    ----------
        - data (pd.DataFrame): Dataset of texts for tokenization
                               Must contain `language` and `text` columns, 
                               where both columns are string-type. 
        - tokenizer (tokenizer): model/tokenizer 
          (optional, defaults to `meta-llama/Llama-3.2-1B-Instruct`)
        - visualize (bool): if `True`, provides side-by-side boxplots of fertilities by language

    Returns
    -------
        - scored (pd.DataFrame): DataFrame with `language`, `corpus`, `fertility`, and `tokens` columns
    
    """

    languages = list(data['language'].unique())
    language_corpora = {} # {'language1':'text1', 'language2', 'text2', etc.}
    fertility_scores = {} # {'language1':'score1', 'language2', 'score2', etc.}
    tokens = {} # {'language1':'tokens1', 'language2', 'tokens2', etc.}

    for language in languages:
        text = data[data['language'] == language]['text']
        corpus = " ".join(text)

        fertility_score, tokenized = token_score(corpus, tokenizer)

        language_corpora[language] = corpus
        fertility_scores[language] = fertility_score
        tokens[language] = tokenized

    scored = pd.DataFrame({'language': pd.Series(languages),
                           'corpus': pd.Series(language_corpora.values()),
                           'fertility': pd.Series(fertility_scores.values()),
                           'tokens': pd.Series(tokens.values())})
    
    # if visualize==True:                                               # for later
    
    return scored 


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# Parity 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def get_parities(data, 
                 tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct'), 
                 visualize=False):
    """ 
    Get parity scores and tokens for a dataset of texts in different languages. 
    
    Parameters
    ----------
        - data (pd.DataFrame): Dataset of texts for tokenization
                               Must contain `language` and `text` columns, 
                               where both columns are string-type. 
        - tokenizer (tokenizer): model/tokenizer 
          (optional, defaults to `meta-llama/Llama-3.2-1B-Instruct`)
        - visualize (bool): if `True`, provides side-by-side boxplots of parities by language

    Returns
    -------
        - scored (pd.DataFrame): DataFrame with `parity` and `tokens` columns added to `data`
    
    """
    parity_scores = []
    tokens = []
    for row_index in range(len(data)):
        text = data.loc[row_index, 'text']
        parity_score, tokenized = token_score(text, tokenizer)
        parity_scores.append(parity_score)
        tokens.append(tokenized)

    scored = data.copy()
    scored['parity'] = pd.Series(parity_scores)
    scored['tokens'] = pd.Series(tokens)

    if visualize==True:
        languages = list(scored['language'].unique())
        parities_by_language = [scored[scored['language']==lang]['parity'] for lang in languages]

        plt.figure(figsize=(12, 6))
        boxprops = dict(facecolor='gold', color='black') 
        medianprops = dict(color='green', linewidth=2)  

        plt.boxplot(
            parities_by_language,
            labels=languages,
            vert=True,
            patch_artist=True,
            boxprops=boxprops,
            medianprops=medianprops,
            showfliers=True
        )

        for i, data in enumerate(parities_by_language, start=1):
            mean = np.mean(data)
            plt.plot(i, mean, 'g*', markersize=12, label='Mean' if i == 1 else "")  
        plt.legend(loc='upper right')

        # plt.xticks(rotation=45, ha='right')
        plt.title('Parity by Language')
        plt.xlabel('Language')
        plt.ylabel('Parity Score')
        plt.show()

    return scored 

