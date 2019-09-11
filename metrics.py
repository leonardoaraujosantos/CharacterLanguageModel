import editdistance as lev_distance
from jiwer import wer
import numpy as np
# pip install -U nltk
from nltk.metrics import distance

def cer(target, predictions):
    """Computes the Character Error Rate (CER).
    CER is defined as the edit distance between the two given strings.
    Args:
      decode: a string of the decoded output.
      target: a string for the ground truth label.
    Returns:
      A float number denoting the CER for the current sentence pair.
    """    
    return distance.edit_distance(predictions, target)

def wer(target, predictions):
    """Computes the Word Error Rate (WER).
    WER is defined as the edit distance between the two provided sentences after
    tokenizing to words.
    Args:
      decode: string of the decoded output.
      target: a string for the ground truth label.
    Returns:
      A float number for the WER of the current decode-target pair.
    """    
    # Map each word to a new char.
    words = set(predictions.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_decode = [chr(word2char[w]) for w in predictions.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_decode), ''.join(new_target))



def levenshtein_score(targets, predictions):
    len_targets = len(targets)
    distance_total = 0
    for idx, target in enumerate(targets):
        try:
            prediction = predictions[idx]
        except:
            prediction = ''
        distance = lev_distance.eval(prediction, target)
        distance_total += distance
    return distance_total / len_targets