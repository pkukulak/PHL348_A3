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
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imshow
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import tensorflow as tf

from load_data import ACT

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

N_HID=300
NUM_TARGS=6

################################################################################

net_data = load("bvlc_alexnet.npy").item()

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


def run_part_5():
    ################################################################################
    #Read Image

    x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
    i = x_dummy.copy()
    img = imresize(((imread("cropped/female/bracco7.jpg")[:,:,:3]).astype(float32)), (227,227,3))
    i[0,:,:,:] = img
    plt.imshow(img)
    plt.show()

    i = i-mean(i)
    #x = tf.Variable(i)
    x = tf.placeholder(float32, i.shape)

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

    # So now conv4 is what we input to our network.
    # Conv4 is a Tensor object.

    ################################################################################
    # Extended Network
    ################################################################################
    M = 384*13*13
    W0 = tf.Variable(tf.random_normal([M, N_HID], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([N_HID], stddev=0.01))

    # Connect conv4 to a 300 hidden units and take tanh.
    conv5 = tf.nn.relu(conv4)
    conv4_flat = tf.reshape(conv4, [1, -1])
    h1 = tf.nn.xw_plus_b(conv4_flat, W0, b0)
    tan1 = tf.nn.tanh(h1)

    # Connect 300 hidden units to 6 output units
    W1 = tf.Variable(tf.random_normal([N_HID, NUM_TARGS], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([NUM_TARGS], stddev=0.01))
    out = tf.nn.xw_plus_b(tan1, W1, b1)

    # Take the softmax of the output.
    prob = tf.nn.softmax(out)

    # Take only the positive parts of the gradients.
    # Note: we take the gradient wrt to the layer before
    # the softmax. Taking it on the softmaxed layer would
    # sometimes produce 0-gradients. This was suggested on piazza.
    gradients = tf.nn.relu(tf.gradients(out,x)[0])

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # Note: We are only taking grads
    grads = sess.run(gradients, feed_dict={x:i})

    ################################################################################
    # Print out the normalized gradients.
    ################################################################################
    normalized_grads = grads[0]*(1/np.amax(grads[0]))
    plt.imshow(normalized_grads)
    plt.show()

    print("Done Part 5. Exiting...")


