from pylab import *
from hashlib import sha256
from scipy.ndimage import filters
from scipy.misc import imread, imresize, imsave
import os
import re
import time
import urllib
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

TRAIN_SIZE = 70
VALID_SIZE = 10
TEST_SIZE = 15
IMG_DIM = (60, 60)
ALEX_DIM = (227, 227)
ALEX_FLAT_DIM = ALEX_DIM[0] * ALEX_DIM[1]
FLAT_IMG_DIM = IMG_DIM[0] * IMG_DIM[1]

MALE_ACT = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
FEMALE_ACT = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
ACT = MALE_ACT + FEMALE_ACT
NUM_TARGS = len(ACT)

ACT_TO_ID = {'butler': 0, 'radcliffe': 1, 'vartan': 2,
             'bracco': 3, 'gilpin': 4, 'harmon': 5}

MALE_FOLDER = 'cropped/male/'
FEMALE_FOLDER = 'cropped/female'

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def encode_one_hot(labels):
    ''''
    Return a n x k matrix, where n is the number of data points
    and k is the number of classes. This matrix is the one-hot
    encoding of the input labels.
    '''
    n = labels.size
    encoded = np.zeros((n, NUM_LABELS))
    encoded[np.arange(n), labels.astype(int)] = 1.0
    return encoded.astype(float64)

def rgb2gray(rgb):
    '''Author: Michael Guerzhoy
    Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    if len(rgb.shape) == 2:
        return rgb
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255.0

def process_image(uncropped, cropped, dims):
    '''
    Using dims, crop and grayscale the image, stored at uncropped. 
    Then, save it in a folder containing cropped images.
    '''
    image = imread(uncropped)
    x1, y1 = int(dims[0]), int(dims[1])
    x2, y2 = int(dims[2]), int(dims[3])
    cropped_image = image[y1:y2, x1:x2]
    imsave(cropped, cropped_image)

def train_valid_test_split(data):
    '''
    Return a training set, a validation set, a test set,
    and all of their respective targets. All are obtained from
    the input data.
    '''
    data = data[data[:,-1].argsort()]
    prev = 0

    N, M = data.shape
    targets, target_indices = np.unique(data[:, -1], return_index=True)
    target_indices = np.append(target_indices, N)
    num_targets = targets.size
    train_in = np.empty([TRAIN_SIZE * num_targets, M-1], dtype=data.dtype)
    train_targ = np.empty([TRAIN_SIZE * num_targets, 1], dtype=data.dtype)

    valid_in = np.empty([VALID_SIZE * num_targets, M-1], dtype=data.dtype)
    valid_targ = np.empty([VALID_SIZE * num_targets, 1], dtype=data.dtype)

    test_in, test_targ = get_test_data(data, targets, target_indices)

    for i in xrange(num_targets):
        train_range = xrange(TRAIN_SIZE * i, TRAIN_SIZE * (i + 1))
        valid_range = xrange(VALID_SIZE * i, VALID_SIZE * (i + 1))
        test_start = target_indices[i+1] - TEST_SIZE

        no_test = data[prev:test_start]
        np.random.shuffle(no_test)
        t_in = no_test[:TRAIN_SIZE, :M-1]
        t_targ = no_test[:TRAIN_SIZE, M-1].reshape(-1, 1)
        train_in[train_range] = t_in
        train_targ[train_range] = t_targ

        v_in = no_test[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE, :M-1]
        v_targ = no_test[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE, M-1].reshape(-1, 1)
        valid_in[valid_range] = v_in
        valid_targ[valid_range] = v_targ

        prev = target_indices[i+1] + 1

    return (train_in.astype(np.float), train_targ, 
            valid_in.astype(np.float), valid_targ, 
              test_in.astype(np.float), test_targ)

def split_to_lower(s):
    '''
    Return the last name of s in lowercase letters.
    '''
    return s.split()[1].lower()

def get_batches(X, Y, batch_size):
    '''
    Return a random data sample of size n from data X
    and their respective labels from Y.
    '''
    n_data, m_data = X.shape
    shuffled_inds = np.random.permutation(n_data)
    shuffled_data = X[shuffled_inds]
    shuffled_targ = Y[shuffled_inds]
    partitioned_data = np.vsplit(shuffled_data, n_data//batch_size)
    partitioned_targ = np.vsplit(shuffled_targ, n_data//batch_size)
    return partitioned_data, partitioned_targ

def encode_one_hot(labels):
    ''''
    Return a n x k matrix, where n is the number of data points
    and k is the number of classes. This matrix is the one-hot
    encoding of the input labels.
    '''
    n = labels.size
    encoded = np.zeros((n, NUM_TARGS))
    encoded[np.arange(n), labels.astype(int)] = 1.0
    return encoded.astype(float64)

def load_data(folder):
    '''
    Load all of the photos located in the given folder
    into a numpy array with labels appended to each datapoint.
    '''
    gray_data = np.empty([1, FLAT_IMG_DIM + 1])
    color_data = np.empty([1, ALEX_DIM[0], ALEX_DIM[1], 3])
    targets = np.empty([])
    files = [f for f in os.listdir(folder) if re.search('\d', f)]
    actors = map(split_to_lower, ACT)
    for filename in files:
        d = re.search("\d", filename)
        identity = filename[:d.start()]
        if identity not in actors:
            continue
        try:
            id = ACT_TO_ID[identity]
            img = imread(folder + filename)

            color_img = imresize(img, ALEX_DIM)
            color_img = color_img.reshape((1,) + color_img.shape)
            color_data = np.append(color_data, color_img, axis=0)

            gray_img = imresize(img, IMG_DIM)
            gray_img = rgb2gray(gray_img).reshape(1, FLAT_IMG_DIM)
            gray_img = np.append(gray_img, id).reshape(1, FLAT_IMG_DIM + 1)
            gray_data = np.append(gray_data, gray_img, axis=0)

            targets = np.append(targets, id)
        except ValueError:
            print "{} ill-formatted. Deleting.".format(filename)
            os.remove(folder + filename)
    return gray_data[1:], color_data[1:], targets.reshape(-1, 1)

def get_test_data(data, targets, target_indices):
    '''
    Returning a test set and corresponding targets.
    Distinct from producing training and validation sets
    because test data is selected deterministically.
    '''
    prev = 0
    N, M = data.shape
    num_targets = targets.size

    test_in = np.empty([TEST_SIZE * num_targets, M-1], dtype=data.dtype)
    test_targ = np.empty([TEST_SIZE * num_targets, 1], dtype=data.dtype)

    for i in xrange(num_targets):
        test_range = xrange(TEST_SIZE * i, TEST_SIZE * (i+1))
        test_start = target_indices[i+1] - TEST_SIZE
        test_end = target_indices[i+1]
        curr_test = data[test_start:test_end, :M-1]
        test_in[test_range] = curr_test
        test_targ[test_range] = np.tile(targets[i], TEST_SIZE).reshape(-1, 1)

        prev = target_indices[i+1] + 1

    return test_in, test_targ

def download_data():
    '''
    Download and save all data from the internet.
    '''
    testfile = urllib.URLopener()
    for a in MALE_ACT:
        name = a.split()[1].lower()
        i = 0
        print a
        for line in open("subset_actors.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                crop_dims = line.split()[5].split(',')
                file_hash = line.split()[-1]
                uncropped_fn = "uncropped/male/"+filename
                cropped_fn = "cropped/male/"+filename
                timeout(testfile.retrieve, (line.split()[4], uncropped_fn), {}, 30)

                if not os.path.isfile("uncropped/male/"+filename):
                    continue
                try:
                    computed_hash = sha256(open(uncropped_fn, 'rb').read())
                    if computed_hash.hexdigest() == file_hash:
                        process_image(uncropped_fn, cropped_fn, crop_dims)
                        print filename
                        i += 1
                    else:
                        print "Hash mismatch : {}".format(filename)
                    os.remove(uncropped_fn)
                except:
                    print "Error occured : {}".format(filename)

    for a in FEMALE_ACT:
        name = a.split()[1].lower()
        i = 0
        print a
        for line in open("subset_actresses.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                crop_dims = line.split()[5].split(',')
                file_hash = line.split()[-1]
                uncropped_fn = "uncropped/female/"+filename
                cropped_fn = "cropped/female/"+filename
                timeout(testfile.retrieve, (line.split()[4], uncropped_fn), {}, 15)

                if not os.path.isfile("uncropped/female/"+filename):
                    continue
                try:
                    computed_hash = sha256(open(uncropped_fn, 'rb').read())
                    if computed_hash.hexdigest() == file_hash:
                        process_image(uncropped_fn, cropped_fn, crop_dims)
                        print filename
                        i += 1
                    else:
                        print "Hash mismatch : {}".format(filename)
                    os.remove(uncropped_fn)
                except:
                    print "Error occured : {}".format(filename)

if __name__ == "__main__":
    if not os.path.exists("cropped/"):
        os.makedirs("cropped/")
        if not os.path.exists("cropped/male/"):
            print "Creating directory ./cropped/male/"
            os.makedirs("cropped/male")
        if not os.path.exists("cropped/female/"):
            print "Creating directory ./cropped/female/"
            os.makedirs("cropped/female")

    if not os.path.exists("uncropped/"):
        os.makedirs("uncropped/")
        if not os.path.exists("uncropped/male/"):
            print "Creating directory ./uncropped/male/"
            os.makedirs("uncropped/male/")
        if not os.path.exists("uncropped/female"):
            print "Creating directory ./uncropped/female/"
            os.makedirs("uncropped/female")

    download_data()
