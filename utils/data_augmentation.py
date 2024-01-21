
import random
import numpy as np

def generate_typo_errors(text, number_typos = 3):
    len_text = len(text)
    random_typos = random.sample(range(len_text),number_typos)
    random_typos = np.sort(random_typos)[::-1]
    new_text = text
    for i in random_typos:
        new_text = new_text[:i] + new_text[i + 1:]
    return new_text