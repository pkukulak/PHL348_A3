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

MALE_ACT = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
FEMALE_ACT = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']

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
                uncropped_filename = "uncropped/male/"+filename
                cropped_filename = "cropped/male/"+filename
                timeout(testfile.retrieve, (line.split()[4], uncropped_filename), {}, 15)

                if not os.path.isfile("uncropped/male/"+filename):
                    continue
                try:
                    computed_hash = sha256(open(uncropped_filename, 'rb').read())
                    if computed_hash.hexdigest() == file_hash:
                        process_image(uncropped_filename, cropped_filename, crop_dims)
                        i += 1
                    else:
                        print "Hash mismatch : {}".format(filename)
                    os.remove(uncropped_filename)
                except:
                    print "Error occured : {}".format(filename)

    for a in FEMALE_ACT:
        name = a.split()[1].lower()
        i = 0
        print a
        for line in open("subset_actresses.txt"):
            if a in line:
                print filename
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                crop_dims = line.split()[5].split(',')
                file_hash = line.split()[-1]
                uncropped_filename = "uncropped/female/"+filename
                cropped_filename = "cropped/female/"+filename
                timeout(testfile.retrieve, (line.split()[4], uncropped_filename), {}, 15)

                if not os.path.isfile("uncropped/female/"+filename):
                    continue
                try:
                    computed_hash = sha256(open(uncropped_filename, 'rb').read())
                    if computed_hash.hexdigest() == file_hash:
                        process_image(uncropped_filename, cropped_filename, crop_dims)
                        i += 1
                    else:
                        print "Hash mismatch : {}".format(filename)
                    os.remove(uncropped_filename)
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
