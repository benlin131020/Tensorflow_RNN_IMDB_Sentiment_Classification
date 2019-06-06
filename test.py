from load_imdb import IMDBData
import numpy as np
import tensorflow as tf

NUM_TRAIN_DATA = 25000
NUM_TEST_DATA = 25000
BATCH_SIZE = 100
LEARNING_RATE = 0.01
EPOCHS = 10
NUM_BATCHES = NUM_TRAIN_DATA // BATCH_SIZE

dataset = IMDBData()
(x_train, y_train), (x_test, y_test), (seqlen_train, seqlen_test), (seq_max_len) = dataset.get_all()
print(y_test.shape, y_train.shape)

x = tf.placeholder(tf.float32, [None, seq_max_len, 1], name='x')
y = tf.placeholder(tf.float32, [None, 2], name='y')
a = x + 1
b = y + 1
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:
    
    sess.run(init)
    for epoch in range(EPOCHS):
        for i in range(NUM_BATCHES):
            print("epoch:", epoch, "batch:", i)
            batch_x, batch_y, batch_seqlen = dataset.next_batch(BATCH_SIZE)
            c, d = sess.run([a, b], feed_dict={x: batch_x, y: batch_y})

        e, f = sess.run([a, b], feed_dict={x: x_train, y: y_train})
        g, h = sess.run([a, b], feed_dict={x: x_test, y: y_test})
    
