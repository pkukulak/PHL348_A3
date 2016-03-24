################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from scipy.ndimage import filters
import urllib
from numpy import random
from random import choice

from load_data import train_valid_test_split, get_batches, train_valid_test_split_part2, load_colored_data, get_partition
from load_data import encode_one_hot

import tensorflow as tf

# Copied directly from starter code.
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())


def run_alex_net(i):
    ''' Run AlexNet on image(s) i, up to the conv4 layer. 
        Return the activations.
    '''

    x = tf.placeholder(tf.float32, i.shape) 

    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    output = sess.run(conv4, feed_dict={x:i})
    return output


def run_part_2():
    ################################################################################
    # Load up the data.
    ################################################################################
    N_HID = 300
    BATCH_SIZE = 1 # recommended by Guerzhoy himself.
    NUM_TARGS = 6
    NUM_ITERS = 10000 # Change this to 10k and rerun.

    # Load the colored data this time.
    print("Loading colored data...")
    data, targets = load_colored_data()
    print("Done loading data.")

    # Load the pretrained network for Alexnet.
    net_data = load("bvlc_alexnet.npy").item()

    ################################################################################
    # Get the activations by running alexnet over our data.
    ################################################################################
    print("Computing activations")
    N, M = data.shape
    activations = run_alex_net(data.reshape((N, 227,227,3)).astype(float32))
    activations = activations /np.amax(activations) # Normalize the activations

    ################################################################################
    # Input the activations into our training data now.
    ################################################################################

    # Partition the data up (see load_data.py).
    (train_in, train_t,
    valid_in, valid_t,
    test_in, test_t) = train_valid_test_split_part2(activations, targets)

    train_in = train_in.reshape((420, 13*13*384))
    valid_in = valid_in.reshape((60, 13*13*384))
    test_in = test_in.reshape((90, 13*13*384))

    train_y = encode_one_hot(train_t.T)
    valid_y = encode_one_hot(valid_t.T)
    test_y = encode_one_hot(test_t.T)


    # Define our network.
    # 384*13*13 inputs (the activations), 300 hidden units, 6 outputs.
    # Tanh on hidden layers, softmax on output layer.
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


    lam = 0.0000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(NLL)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Arrays for storing learning curves.
    test_accs = []
    test_lps = []

    valid_accs = []
    valid_lps = []

    train_accs = []
    train_lps = []

    # Train the network.
    print("Training...")
    for i in xrange(NUM_ITERS):
        batches_in, batches_t = get_batches(train_in, train_t, BATCH_SIZE)
        batch_in, batch_t = choice(zip(batches_in, batches_t))

        batch_in = batch_in.reshape(-1, M)
        batch_t = encode_one_hot(batch_t.T)
        sess.run(train_step, feed_dict={x: batch_in, y_: batch_t})

        # Check performance every 50 iterations.
        if i % 50 == 0:
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

    #####################################################################
    #  Plot the data
    #####################################################################
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

    # Plot the accuracy.
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(train_accs, 'b', valid_accs, 'r', test_accs, 'g')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc=4)
    plt.show()


    print("Finished part 2. Exiting...")


