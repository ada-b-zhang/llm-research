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
    - `fertility`
    - `get_fertilities`
    - 

""" 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# Fertility 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def fertility(text, tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')):
    """ 
    Get the fertility score and tokens for a given text.

    Parameters
    ----------
        - text (str): text for tokenization
        - tokenizer (tokenizer): model/tokenizer 
          (optional, defaults to `meta-llama/Llama-3.2-1B-Instruct`)

    Returns
    -------
        - fertility (float): fertility score
        - tokenized (list): list of tokens 
    """ 
    tokenized = tokenizer.tokenize(text) 
    num_words = len(text.split())

    fertility = len(tokenized) / num_words if num_words > 0 else 999999 
    return fertility, tokenized

def get_fertilities(data, 
                    tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct'), 
                    visualize=False, 
                    measure='average'):
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
        - scored (pd.DataFrame): DataFrame with `fertility` and `tokens` columns added to `data`

    
    """
    fertility_scores = []
    tokens = []
    for row_index in range(len(data)):
        text = data.loc[row_index, 'text']
        fertility_score, tokenized = fertility(text, tokenizer)
        fertility_scores.append(fertility_score)
        tokens.append(tokenized)

    scored = data.copy()
    scored['fertility'] = pd.Series(fertility_scores)
    scored['tokens'] = pd.Series(tokens)


    if visualize==True:
        languages = list(scored['language'].unique())
        fertilities_by_language = [scored[scored['language']==lang]['fertility'] for lang in languages]

        plt.figure(figsize=(12, 6))
        boxprops = dict(facecolor='gold', color='black') 
        medianprops = dict(color='green', linewidth=2)  

        plt.boxplot(
            fertilities_by_language,
            labels=languages,
            vert=True,
            patch_artist=True,
            boxprops=boxprops,
            medianprops=medianprops,
            showfliers=True
        )

        for i, data in enumerate(fertilities_by_language, start=1):
            mean = np.mean(data)
            plt.plot(i, mean, 'g*', markersize=12, label='Mean' if i == 1 else "")  
        plt.legend(loc='upper right')

        # plt.xticks(rotation=45, ha='right')
        plt.title('Fertility by Language')
        plt.xlabel('Language')
        plt.ylabel('Fertility Score')
        plt.show()

    return scored 


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# Parity 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""