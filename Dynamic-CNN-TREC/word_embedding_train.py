# -*- coding: UTF-8 -*-
# python3.5(tensorflow)：C:\Users\Dr.Du\AppData\Local\conda\conda\envs\tensorflow\python.exe
# python3.6：C:\ProgramData\Anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2018/5/9 20:50
# @Author  : tuhailong

from gensim.models import word2vec
import re
import numpy as np
from tensorflow.contrib import learn

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " ", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"`", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"''", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_w2v_model():
    sentence = []
    with open("./data/train", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = clean_str(line)
            line = line.split(":")[1:]
            line = ' '.join(line).strip()
            word_list = line.split()
            sentence.append(word_list)

    with open("./data/test", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = clean_str(line)
            line = line.split(":")[1:]
            line = ' '.join(line).strip()
            word_list = line.split()
            sentence.append(word_list)
    #print(len(sentence))
    model = word2vec.Word2Vec(sentence, size=32, window=5, min_count=0, workers=10)
    model.wv.save_word2vec_format("./w2v.model.txt", binary=False)

def loadWord2Vec(filename):
    vocab = []
    embd = []
    fr = open(filename, 'r')
    line = fr.readline()
    word_dim = int(line.split(' ')[1])
    embd.append([0]*word_dim)
    for line in fr:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print("loaded word2vec")
    fr.close()
    return vocab, embd

def get_index(vocab, raw_input, max_len):
    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_len)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    #print(raw_input)
    input_x = np.array(list(vocab_processor.transform(raw_input)))
    return input_x

def load_data_and_labels():
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    folder_prefix = 'data/'
    x_train = list(open(folder_prefix+"train").readlines())
    x_test = list(open(folder_prefix+"test").readlines())
    test_size = len(x_test)
    x_text = x_train + x_test
    x_text = [clean_str(sent) for sent in x_text]
    y = [s.split(':')[0] for s in x_text]
    x_text = [s.split(" ")[1:] for s in x_text]
    x_text = [" ".join(s).strip() for s in x_text]
    # Generate labels
    all_label = dict()
    for label in y:
        if not label in all_label:
            all_label[label] = len(all_label) + 1
    one_hot = np.identity(len(all_label))
    y = [one_hot[all_label[label]-1] for label in y]
    return [x_text, y, test_size]

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

def load_data():
    #get_w2v_model()
    vocab, embd = loadWord2Vec("./w2v.model.txt")
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    print(len(embedding))
    x_text, y, test_size = load_data_and_labels()
    max_len = max(len(x.split()) for x in x_text)
    print(x_text)
    input_x = get_index(vocab, x_text, max_len)
    #print(get_index(vocab, ['spock'], 4))
    #print(embedding[0])
    y = np.array(y)
    return input_x, y, vocab, test_size, embedding

