import tensorflow as tf

class DCNN():
    def __init__(self, batch_size, sentence_length, num_filters, embed_size, top_k):
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.top_k = top_k

    def conv_layer(self, x, w, b):
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        with tf.name_scope("per_dim_conv"):
            for i in range(len(input_unstack)):
                #print(w_unstack[i].shape[0])
                pad_size = int((int(w_unstack[i].shape[0])-1)/2)
                #print(pad_size)
                if w_unstack[i].shape[0] % 2 == 0:
                    input_pad = tf.pad(input_unstack[i], [[0, 0], [pad_size+1, pad_size], [0, 0]], mode='CONSTANT')
                else:
                    input_pad = tf.pad(input_unstack[i], [[0, 0], [pad_size, pad_size], [0, 0]], mode="CONSTANT")
                conv = tf.nn.conv1d(input_pad, w_unstack[i], stride=1, padding="SAME")
                conv = tf.nn.tanh(conv + b_unstack[i])#[batch_size, k1+ws2-1, num_filters[1]]
                convs.append(conv)
            conv = tf.stack(convs, axis=2)
            #[batch_size, k1+ws-1, embed_size, num_filters[1]]
        return conv

    def fold_k_max_pooling(self, x, k):
        input_unstack = tf.unstack(x, axis=2)
        out = []
        with tf.name_scope("fold_k_max_pooling"):
            for i in range(0, len(input_unstack), 2):
                fold = tf.add(input_unstack[i], input_unstack[i+1])#[batch_size, k1, num_filters[1]]
                conv = tf.transpose(fold, perm=[0, 2, 1])
                values = tf.nn.top_k(conv, k, sorted=False).values #[batch_size, num_filters[1], top_k]
                values = tf.transpose(values, perm=[0, 2, 1])
                out.append(values)
            fold = tf.stack(out, axis=2)#[batch_size, k2, embed_size/2, num_filters[1]]
        return fold

    def full_connect_layer(self, x, w, b, wo, dropout_keep_prob):
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(x, w) + b)
            h = tf.nn.dropout(h, dropout_keep_prob)
            o = tf.matmul(h, wo)
        return o

    def run(self, sent, W1, W2, b1, b2, top_k, Wh, bh, Wo, dropout_keep_prob):
        conv1 = self.conv_layer(sent, W1, b1)
        k = int(int(sent.shape[1])/2)
        conv1 = self.fold_k_max_pooling(conv1, k)
        conv2 = self.conv_layer(conv1, W2, b2)
        fold = self.fold_k_max_pooling(conv2, top_k)
        fold_flatten = tf.reshape(fold, [-1, int(top_k*self.embed_size*self.num_filters[1]/4)])
        print(fold_flatten.get_shape())
        out = self.full_connect_layer(fold_flatten, Wh, bh, Wo, dropout_keep_prob)
        return out
