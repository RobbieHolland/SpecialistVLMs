import random
import torch
import numpy as np
import re

def list_or(elements, randomise=True):
    return list_and(elements, joiner='or', randomise=randomise)

def list_and(elements, joiner='and', randomise=True):
    if isinstance(elements, str):
        return elements
    elements = [b for b in elements if valid_variable(b)]
    if elements is None or len(elements) == 0:
        return ''
    if randomise:
        elements = random.sample(elements, len(elements))
    return ', '.join(elements[:-1]) + f" {joiner} {elements[-1]}" if len(elements) > 1 else elements[0]

def valid_variable(var):
    if var is None or var == 'nan' or var is np.nan or var is float('nan'):
        return False
    if isinstance(var, (float, np.float64)) and np.isnan(var):
        return False
    if isinstance(var, np.ndarray) and np.isnan(var).any():
        return False
    if torch.is_tensor(var) and torch.isnan(var).any():
        return False
    return True

def parse_qa(text, blacklist_words=['description', 'mention', 'specified']):
    if not valid_variable(text):
        return None
    # Split based on integer-dot format
    chunks = re.split(r'^\d+\.', text, flags=re.M)
    
    qa_list = []

    for chunk in chunks[1:]:
        lines = [line.strip() for line in chunk.split('\n') if line.strip()]
        if len(lines) == 2:
            question, answer = lines
            # Removing any labels like "USER:", "ASSISTANT:", etc., just to get pure content
            question = re.split(r'(Q|USER):', question)[-1].strip()
            answer = re.sub(r'(ASS|ANS)\S*:?', '', answer).strip()
            answer = re.sub(r'^A:', '', answer).strip()

            if any([w in answer for w in blacklist_words]):
                continue

            qa_list.append({
                'Question': question,
                'Answer': answer
            })

    if len(qa_list) == 0:
        return None

    return qa_list
