from load_data import *
import tensorflow as tf

N_HID = 300
BATCH_SIZE = 60
NUM_TARGS = 6
NUM_ITERS = 10000

if __name__ == '__main__':
    male_data          = load_data('cropped/male/')
    female_data        = load_data('cropped/female/')
    data               = np.vstack((male_data, female_data))
    (train_in, train_t,
    valid_in, valid_t,
    test_in, test_t)    = train_valid_test_split(data)

    _, M = train_in.shape

    # Tensorflow variables.
    x  = tf.placeholder(tf.float32, [None, M])

    # Hidden layer weights and bias.
    W0 = tf.Variable(tf.random_normal([M, N_HID], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([N_HID], stddev=0.01))

    # Output layer weights. 
    W1 = tf.Variable(tf.random_normal([N_HID, NUM_TARGS], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([NUM_TARGS], stddev=0.01))

    layer1 = tf.nn.tanh(tf.matmul(x, W0) + b0)
    layer2 = tf.matmul(layer1, W1) + b1

    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, NUM_TARGS])

    lam = 0.00000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

    train_step = tf.train.GradientDescentOptimizer(0.00005).minimize(NLL)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    for i in xrange(NUM_ITERS):
        batches_in, batches_t = get_batches(train_in, train_t, BATCH_SIZE)
        for batch_in, batch_t in zip(batches_in, batches_t):
            batch_in = batch_in.reshape(-1, M)
            batch_t = encode_one_hot(batch_t)
            sess.run(train_step, feed_dict={x: batch_in, y_: batch_t})
            if i % 10 == 0:
                print "i=",i
                test_x = test_in.reshape(-1, M)
                test_y = encode_one_hot(test_t)
                print "Test:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y})

                print "Train:", sess.run(accuracy, feed_dict={x: batch_in, y_: batch_t})
                print "Penalty:", sess.run(decay_penalty)
                
