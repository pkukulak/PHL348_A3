from load_data import *
import tensorflow as tf

N_HID = 300
BATCH_SIZE = 60
NUM_TARGS = 6
NUM_ITERS = 5000

if __name__ == '__main__':
    male_data, _, __   = load_data('cropped/male/')
    female_data, _, __ = load_data('cropped/female/')
    data               = np.vstack((male_data, female_data))
    (train_in, train_t,
    valid_in, valid_t,
    test_in, test_t)    = train_valid_test_split(data)
    _, M = train_in.shape
    train_y = encode_one_hot(train_t.T)
    test_y = encode_one_hot(test_t.T)
    valid_y = encode_one_hot(valid_t.T)
    
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

    lam = 0.0000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

    train_step = tf.train.GradientDescentOptimizer(0.00005).minimize(NLL)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    test_accs = []
    test_lps = []

    valid_accs = []
    valid_lps = []

    train_accs = []
    train_lps = []

    for i in xrange(NUM_ITERS):
        batches_in, batches_t = get_batches(train_in, train_t, BATCH_SIZE)
        batch_in, batch_t = random.choice(zip(batches_in, batches_t))
        batch_in = batch_in.reshape(-1, M)
        batch_t = encode_one_hot(batch_t.T)
        sess.run(train_step, feed_dict={x: batch_in, y_: batch_t})
        if i % 10 == 0:
            print "i=",i
            valid_x = valid_in.reshape(-1, M)
            valid_acc = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            valid_accs += [valid_acc]
            valid_lp = sess.run(NLL, feed_dict={x: valid_x, y_: valid_y})
            valid_lps += [valid_lp]

            test_x = test_in.reshape(-1, M)
            test_acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            test_accs += [test_acc]
            test_lp = sess.run(NLL, feed_dict={x: test_x, y_: test_y})
            test_lps += [test_lp]

            train_acc = sess.run(accuracy, feed_dict={x: train_in, y_: train_y})
            train_accs += [train_acc]
            train_lp = sess.run(NLL, feed_dict={x: train_in, y_: train_y})
            train_lps += [train_lp]
            
            print 'TEST ACCURACY  = ', test_acc
            print 'VALID ACCURACY = ', valid_acc
            print 'TRAIN ACCURACY = ', train_acc
           

    red_patch = mpatches.Patch(color='red', label='Validation')
    blue_patch = mpatches.Patch(color='blue', label='Training')
    green_patch = mpatches.Patch(color='green', label='Test')

    # Plot the learning curves.
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.plot(train_lps, 'b', valid_lps, 'r', test_lps, 'g')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc=1)
    plt.show()

    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(train_accs, 'b', valid_accs, 'r', test_accs, 'g')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc=4)
    plt.show()

