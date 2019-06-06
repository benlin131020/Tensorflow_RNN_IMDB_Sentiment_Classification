from load_imdb import IMDBData
import numpy as np
import tensorflow as tf

NUM_TRAIN_DATA = 25000
NUM_TEST_DATA = 25000
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10
NUM_BATCHES = NUM_TRAIN_DATA // BATCH_SIZE
NUM_HIDDEN = 64
NUM_CLASSES = 2

def dynamicRNN(x, seqlen, weights, biases, seq_max_len):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, NUM_HIDDEN]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

if __name__ == "__main__":
    dataset = IMDBData()
    (x_train, y_train), (x_test, y_test), (seqlen_train, seqlen_test), (seq_max_len) = dataset.get_all()
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([NUM_HIDDEN, NUM_CLASSES]), name='W')
    }
    biases = {
        'out': tf.Variable(tf.random_normal([NUM_CLASSES]), name='B')
    }
    with tf.name_scope("InputLayer"):
        #shape of input is [batch_size, sequence_length(timesteps), number_features]
        x = tf.placeholder(tf.float32, [None, seq_max_len, 1], name='x')
        y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='y')
        # A placeholder for indicating each sequence length
        seqlen = tf.placeholder(tf.int32, [None])
    with tf.name_scope("DynamicRNN"):
        pred = dynamicRNN(x, seqlen, weights, biases, seq_max_len)
        pred_softmax = tf.nn.softmax(pred)
    with tf.name_scope("Loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
        tf.summary.scalar("cross entropy", cost)
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    with tf.name_scope("Accuracy"):
        correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        test_writer = tf.summary.FileWriter("logs/test")
        merged = tf.summary.merge_all()
        for epoch in range(EPOCHS):
            for i in range(NUM_BATCHES):
                batch_x, batch_y, batch_seqlen = dataset.next_batch(BATCH_SIZE)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            
            #training data's loss and acc
            acc, loss, train_result = sess.run([accuracy, cost, merged], feed_dict={x: x_train, y: y_train, seqlen: seqlen_train})
            train_writer.add_summary(train_result, epoch)
            #testing data's loss and acc
            test_acc, test_loss, test_result = sess.run([accuracy, cost, merged], feed_dict={x: x_test, y: y_test, seqlen: seqlen_test})
            test_writer.add_summary(test_result, epoch)
            print("Epoch%02d:" % (epoch+1), "loss:{:.9f}".format(loss), "accuracy:", acc, "test_loss:{:.9f}".format(test_loss), "test_accuracy:", test_acc)

        print("Optimization Finished!")
