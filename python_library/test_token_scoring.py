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

