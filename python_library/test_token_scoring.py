from transformers import AutoTokenizer
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from token_scoring import *


def test_fertility_valid_text():
    text = 'This is a sentence for testing.'
    expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġsentence', 'Ġfor', 'Ġtesting', '.']

    fertility_score, tokenized = fertility(text)
    expected_fertility = len(tokenized) / len(text.split())

    assert expected_tokens == tokenized
    assert fertility_score == expected_fertility


def test_fertility_empty_text():
    text = "" 
    expected_tokens = []

    fertility_score, tokenized = fertility(text)
    expected_fertility = 999999

    assert expected_tokens == tokenized 
    assert fertility_score == expected_fertility

def test_flan_t5():
    texts = ["The sky is bright today.", 
            "I love classical music.", 
            "Data science is fascinating.", 
            "Could you pass the salt?", 
            "Baroque composers inspire my work.", 
            "What time is the meeting?", 
            "This coffee tastes really good.",
            "Purple is pretty"]

    # Model: google/flan-t5-xxl
    expected_tokens = [['▁The', '▁sky', '▁is', '▁bright', '▁today', '.'],
                        ['▁I', '▁love', '▁classical', '▁music', '.'],
                        ['▁Data', '▁science', '▁is', '▁fascinating', '.'],
                        ['▁Could', '▁you', '▁pass', '▁the', '▁salt', '?'],
                        ['▁Bar', 'o', 'que', '▁composer', 's', '▁inspire', '▁my', '▁work', '.'],
                        ['▁What', '▁time', '▁is', '▁the', '▁meeting', '?'],
                        ['▁This', '▁coffee', '▁tastes', '▁really', '▁good', '.'],
                        ['▁Purple', '▁is', '▁pretty']]

    expected_token_IDs = [[450, 14744, 338, 11785, 9826, 29889],
                            [306, 5360, 14499, 4696, 29889],
                            [3630, 10466, 338, 21028, 262, 1218, 29889],
                            [6527, 366, 1209, 278, 15795, 29973],
                            [2261, 29877, 802, 5541, 414, 8681, 533, 590, 664, 29889],
                            [1724, 931, 338, 278, 11781, 29973],
                            [910, 26935, 260, 579, 267, 2289, 1781, 29889],
                            [15247, 552, 338, 5051]]
    tokens = []
    for text in texts: 
        fertility_score, tokenized = fertility(text, 
                                               tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-xxl'))
        tokens.append(tokenized)

    assert expected_tokens==tokens

