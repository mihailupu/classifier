import numpy as np
import re
import itertools
from collections import Counter

wipoareas = {55: 0, 56: 1, 57: 2, 58: 3, 59: 4, 60: 5, 61: 6, 62: 7, 63: 8, 64: 9, 65: 10, 66: 11, 67: 12, 68: 13,
             69: 14, 70: 15, 71: 16, 72: 17, 73: 18, 74: 19,
             77: 20, 78: 21, 79: 22, 80: 23, 81: 24, 82: 25, 83: 26, 84: 27, 85: 28, 86: 29, 87: 30, 88: 31, 89: 32,
             90: 33, 91: 34}
inv_wipoareas = {v: k for k, v in wipoareas.items()}

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    modified for pantent abstracts a bit, to remove existing \n elements
    """
    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"\t", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_wipo_data_and_labels(data_file):
    """
    loads data from a file that contains tab separated fields:
    newid wipo patnr kind title abstract
    """


    # Load data from files
    x_text = []
    prevPatnr = "BOGUS1"
    yValue = []
    y = []
    with open(data_file, "r", encoding='utf-8' ) as df:
        for line in df:
            data = line.split('\t')
            if len(data) == 6 and data[0] != "newid":
                wipo = int(data[1])
                patnr = data[2]
                text = data[4] + data[5]
                if patnr != prevPatnr:
                    prevPatnr = patnr
                    if len(yValue) > 0:
                        y.append(yValue)
                    x_text.append(clean_str(text.strip()))
                    yValue = [0 for _ in wipoareas]
                    yValue[wipoareas[wipo]] = 1
                else:
                    yValue[wipoareas[wipo]] = 1
    y.append(yValue)
    return [x_text, np.asarray(y)]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]{3,}",
                          re.UNICODE)


def tokenizer(iterator):
  """Tokenizer generator.

  Args:
    iterator: Input iterator with strings.

  Yields:
    array of tokens per each value in the input.
  """
  for value in iterator:
    yield TOKENIZER_RE.findall(value)