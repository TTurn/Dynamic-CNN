import tensorflow as tf

class DCNN():
    def __init__(self, batch_size, embed_size, num_filters, top_k):
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.num_filters = num_filters
        self.top_k = top_k

    def conv_layer(self, x, w, b):
        input_unstack = tf.unstack(x, axis=2)   #input = [batch_size, sentence_len, embedding_len, 1]
        w_unstack = tf.unstack(w, axis=1)      #w = [filter_len, embedding_size, 1, filter_num]
        b_unstack = tf.unstack(b, axis=1)      #b = [fiter_num, embedding_size]
        convs = []
        with tf.name_scope("conv"):
            for i in range(len(input_unstack)):     #embedding_size
                pad_left = int(int(w_unstack[i].shape[0]-1)/2)
                if w_unstack[i].shape[0] % 2 == 1:
                    pad_right = pad_left
                else:
                    pad_right = pad_left+1
                input_unstack_pad = tf.pad(input_unstack[i], [[0, 0], [pad_left, pad_right], [0, 0]], 'CONSTANT')
                conv = tf.nn.tanh(tf.nn.conv1d(input_unstack_pad, w_unstack[i], stride=1, padding="SAME")+b_unstack[i])
                convs.append(conv)
            conv = tf.stack(convs, axis=2)
            #[batch_size, after_filter, embedding_size, num_filter]
        return conv

    def fold_k_max_pool_layer(self, x, num_of_layers, layer_num):
        print("stentence length:", x.shape[1])
        k = int(max(self.top_k, (num_of_layers-layer_num)/float(num_of_layers)*int(x.shape[1])))
        input_unstack = tf.unstack(x, axis=2)      #input = [batch_size, after_filter, embedding_size, num_filter]
        pool_out = []
        with tf.name_scope("fold_pooling"):
            for i in range(0, len(input_unstack), 2):
                fold = tf.add(input_unstack[i], input_unstack[i+1])
                fold = tf.transpose(fold, [0, 2, 1])
                values = tf.nn.top_k(fold, k, sorted=False).values    #k_max_pooling
                values = tf.transpose(values, [0, 2, 1])
                pool_out.append(values)
            fold_pool = tf.stack(pool_out, axis=2)
        return fold_pool

    def full_connection_layer(self, x, w, b, wo, dropout_keep_prob):
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(x, w)+b)
            h = tf.nn.dropout(h, dropout_keep_prob)
            out = tf.matmul(h, wo)
        return out

    def DCNN(self, sent, W1, b1, W2, b2, Wh, bh, Wo, dropout_keep_prob):


        conv1 = self.conv_layer(sent, W1, b1)
        print("conv1_out:", conv1.shape)
        fold_pool1 = self.fold_k_max_pool_layer(conv1, num_of_layers=2, layer_num=1)
        print("pool1_out:", fold_pool1.shape)
        conv2 = self.conv_layer(fold_pool1, W2, b2)
        print("conv2_out:", conv2.shape)
        fold_pool2 = self.fold_k_max_pool_layer(conv2, num_of_layers=2, layer_num=2)
        print("pool2_out:", fold_pool2.shape)
        fold_flattern = tf.reshape(fold_pool2, [-1, int(self.top_k*self.embed_size*self.num_filters[1]/4)])
        print("full_in:", fold_flattern.shape)
        out = self.full_connection_layer(fold_flattern, Wh, bh, Wo, dropout_keep_prob)
        print("out:", out.shape)
        return out

