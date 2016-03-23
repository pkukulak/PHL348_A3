from load_data import *
import tensorflow as tf

BATCH_SIZE = 60
NUM_TARGS = 6
NUM_ITERS = 6000

male_data   = load_data_part1('cropped/male/')
female_data = load_data_part1('cropped/female/')
data        = np.vstack((male_data, female_data))
(train_in, train_t,
valid_in, valid_t,
test_in, test_t)    = train_valid_test_split(data)
_, M = train_in.shape

train_y = encode_one_hot(train_t.T)

test_x = test_in.reshape(-1, M)
test_y = encode_one_hot(test_t.T)

valid_x = valid_in.reshape(-1, M)
valid_y = encode_one_hot(valid_t.T)

def train_nn(n_hid):
    # Tensorflow variables.
    x  = tf.placeholder(tf.float32, [None, M])

    # Hidden layer weights and bias.
    W0 = tf.Variable(tf.random_normal([M, n_hid], stddev=0.01)/100)
    b0 = tf.Variable(tf.random_normal([n_hid], stddev=0.01)/100)

    # Output layer weights. 
    W1 = tf.Variable(tf.random_normal([n_hid, NUM_TARGS], stddev=0.01)/100)
    b1 = tf.Variable(tf.random_normal([NUM_TARGS], stddev=0.01)/100)

    layer1 = tf.nn.tanh(tf.matmul(x, W0) + b0)
    layer2 = tf.matmul(layer1, W1) + b1

    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, NUM_TARGS])

    lam = 0.0003
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
        batch_in, batch_t = random.choice(zip(batches_in, batches_t))
        batch_in = batch_in.reshape(-1, M)
        batch_t = encode_one_hot(batch_t.T)
        sess.run(train_step, feed_dict={x: batch_in, y_: batch_t})
        if i % 10 == 0:
            print "i=",i
            valid_acc = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})

            test_x = test_in.reshape(-1, M)
            test_acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})

            train_acc = sess.run(accuracy, feed_dict={x: train_in, y_: train_y})
            print 'TEST ACCURACY  = ', test_acc
            print 'VALID ACCURACY = ', valid_acc
            print 'TRAIN ACCURACY = ', train_acc

    # Vizzalize sum waits
    feature_1 = sess.run(W0)[:, 543]
    feature_2 = sess.run(W0)[:, 748]

    fig = figure()
    ax = fig.gca()
    heatmap = ax.imshow(feature_1.reshape(60,60), cmap=cm.coolwarm)
    show()
    
    fig = figure()
    ax = fig.gca()
    heatmap = ax.imshow(feature_2.reshape(60, 60), cmap=cm.coolwarm)
    show()

if __name__ == '__main__':
    train_nn(300)
    train_nn(800)

    print("Done Part 3. Exiting...")
