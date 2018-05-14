from collections import Counter
import itertools
import numpy as np
import re

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
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

def load_data_and_labels(data):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    folder_prefix = 'sst_data/'
    filename_sst2 = ['sst2_train_phrases.csv', 'sst2_dev.csv', 'sst2_test.csv']
    filename_sst5 = ['sst5_train_phrases.csv', 'sst5_dev.csv', 'sst5_test.csv']
    if data == 2:
        filename_sst = filename_sst2
    else:
        filename_sst = filename_sst5

    x_train = list(open(folder_prefix+filename_sst[0]).readlines())
    x_dev = list(open(folder_prefix+filename_sst[1]).readlines())
    x_test = list(open(folder_prefix+filename_sst[2]).readlines())
    dev_size = len(x_dev)
    test_size = len(x_test)
    x_text = x_train + x_dev + x_test
    x_text = [clean_str(sent) for sent in x_text]
    y = [s.split(' ')[-1] for s in x_text]
    x_text = [s.split(" ")[:-1] for s in x_text]

    x_text_clip = []
    y_clip = []
    for i in range(len(x_text)-(test_size+dev_size)):
        if len(x_text[i]) >= 5:
            x_text_clip.append(x_text[i])
            y_clip.append(y[i])
    x_text = x_text_clip
    y = y_clip

    # Generate labels
    all_label = dict()
    for label in y:
        if not label in all_label:
            all_label[label] = len(all_label) + 1
    one_hot = np.identity(len(all_label))
    y = [one_hot[all_label[label]-1] for label in y]
    return [x_text, y, test_size, dev_size]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    # vocabulary_inv=['<PAD/>', 'the', ....]
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    # vocabulary = {'<PAD/>': 0, 'the': 1, ',': 2, 'a': 3, 'and': 4, ..}
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data(data):
    """
    Loads and preprocessed data
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, test_size, dev_size = load_data_and_labels(data)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, test_size, dev_size]

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(list(data))
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index > data_size:
                end_index = data_size
                start_index = end_index - batch_size
            yield shuffled_data[start_index:end_index]