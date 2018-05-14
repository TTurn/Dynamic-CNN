import numpy as np
import time
import os
import model
import dataUtils
import tensorflow as tf

num_class = 2
if num_class == 5:
    embed_dim = 48
    ws = [10, 7]
    top_k = 5
    num_filters = [6, 12]
    dev = 300
    batch_size = 512
    n_epochs = 30
    num_hidden = 100
    sentence_length = 53
    lr = 1e-3
    evaluate_every = 100
    checkpoint_every = 100
    num_checkpoints = 5

elif num_class == 2:
    embed_dim = 48
    ws = [7, 5]
    top_k = 4
    num_filters = [6, 14]
    dev = 300
    batch_size = 512
    n_epochs = 50
    num_hidden = 100
    sentence_length = 52
    lr = 1e-3
    evaluate_every = 100
    checkpoint_every = 100
    num_checkpoints = 5

# Load data
print("Loading data...")
x_, y_, vocabulary, vocabulary_inv, test_size, dev_size = dataUtils.load_data(num_class)
print(y_)
# Randomly shuffle data
x_train, x_dev, x_test = x_[:-(test_size+dev_size)], x_[-(test_size+dev_size):-test_size], x_[-test_size:]
y_train, y_dev, y_test = y_[:-(test_size+dev_size)], y_[-(test_size+dev_size):-test_size], y_[-test_size:]

print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))
#--------------------------------------------------------------------------------------#

def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)

sent = tf.placeholder(tf.int64, [None, sentence_length])
y = tf.placeholder(tf.float64, [None, num_class])
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")


with tf.name_scope("embedding_layer"):
    W = tf.Variable(tf.random_uniform([len(vocabulary), embed_dim], -1.0, 1.0), name="embed_W")
    sent_embed = tf.nn.embedding_lookup(W, sent)
    #input_x = tf.reshape(sent_embed, [batch_size, -1, embed_dim, 1])
    input_x = tf.expand_dims(sent_embed, -1)
    #[batch_size, sentence_length, embed_dim, 1]

W1 = init_weights([ws[0], embed_dim, 1, num_filters[0]], "W1")
b1 = tf.Variable(tf.constant(0.1, shape=[num_filters[0], embed_dim]), "b1")

W2 = init_weights([ws[1], int(embed_dim/2), num_filters[0], num_filters[1]], "W2")
b2 = tf.Variable(tf.constant(0.1, shape=[num_filters[1], embed_dim]), "b2")

Wh = init_weights([int(top_k*embed_dim*num_filters[1]/4), num_hidden], "Wh")
bh = tf.Variable(tf.constant(0.1, shape=[num_hidden]), "bh")

Wo = init_weights([num_hidden, num_class], "Wo")

dcnn = model.DCNN(batch_size, sentence_length, num_filters, embed_dim, top_k)
out = dcnn.run(input_x, W1, W2, b1, b2, top_k, Wh, bh, Wo, dropout_keep_prob)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))+tf.contrib.layers.l2_regularizer(0.0005)(W1)+tf.contrib.layers.l2_regularizer(0.0005)(W2)+tf.contrib.layers.l2_regularizer(0.005)(Wh)+tf.contrib.layers.l2_regularizer(0.005)(Wo)
# train_step = tf.train.AdamOptimizer(lr).minimize(cost)

predict_op = tf.argmax(out, axis=1, name="predictions")
with tf.name_scope("accuracy"):
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(out, 1)), tf.float32))
#-------------------------------------------------------------------------------------------#

print('Started training')
with tf.Session() as sess:
    #init = tf.global_variables_initializer().run()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(cost)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cost)
    acc_summary = tf.summary.scalar("accuracy", acc)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        feed_dict = {
            sent: x_batch,
            y: y_batch,
            dropout_keep_prob: 0.5
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cost, acc],
            feed_dict)
        print("TRAIN step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            sent: x_batch,
            y: y_batch,
            dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, cost, acc],
            feed_dict)
        print("VALID step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)
        return accuracy, loss

    batches = dataUtils.batch_iter(zip(x_train, y_train), batch_size, n_epochs)

    # Training loop. For each batch...
    max_acc = 0
    best_at_step = 0
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            acc_dev, _ = dev_step(x_dev, y_dev, writer=dev_summary_writer)
            if acc_dev >= max_acc:
                max_acc = acc_dev
                best_at_step = current_step
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("")
        if current_step % checkpoint_every == 0:
            print('Best of valid = {}, at step {}'.format(max_acc, best_at_step))

    saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
    print('Finish training. On test set:')

    acc, loss = dev_step(x_test, y_test, writer=None)

    print(acc, loss)