import tensorflow as tf
import dataUtil
import numpy as np


def create_record(dev_size):
    print("create_record")
    #writer = tf.python_io.TFRecordWriter("train.tfrecords")

    x_text, y, test_size = dataUtil.load_data_and_labels()
    sentences_padded = dataUtil.pad_sentences(x_text)
    vocabulary, vocabulary_inv = dataUtil.build_vocab(sentences_padded)
    x, y = dataUtil.build_input_data(sentences_padded, y, vocabulary)

    x, x_test = x[:-test_size], x[-test_size:]
    y, y_test = y[:-test_size], y[-test_size:]


    shuffle_indices = np.random.permutation(np.arange(len(y)))
    print("shuffle")
    x_shuffled = np.array(x)[shuffle_indices]
    y_shuffled = np.array(y)[shuffle_indices]
    x_train, x_dev = x_shuffled[:-dev_size], x_shuffled[-dev_size:]
    y_train, y_dev = y_shuffled[:-dev_size], y_shuffled[-dev_size:]
    #print(y_shuffled[:1000])
    return x_dev, y_dev, x_test, y_test
"""
    for i in range(len(y_train)):
        #print(sent)
	    #print(index)
        x_bytes = x_train[i].tobytes()
        y_bytes = y_train[i].tobytes()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_bytes])),
                    "sent": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_bytes]))
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()    
"""




