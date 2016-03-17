from load_data import load_data, get_batches
import tensorflow as tf

N_HID = 350
BATCH_SIZE = 150
NUM_TARGS = 6

if __name__ == '__main__':
    male_data          = load_data('uncropped/male/')
    female_data        = load_data('uncropped/female/')
    data               = np.vstack((male_data, female_data))
    train_in, train_t,
    valid_in, valid_t,
    test_in, test_t    = train_valid_test_split(data)

    N, M = data.shape

    # Tensorflow variables.
    x  = tf.placeholder(tf.float32, [None, M])
    W0 = tf.Variable(tf.random_normal([M, N_HID], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([N_HID], stddev=0.01))
    W1 = tf.Variable(tf.random_normal([N_HID, NUM_TARGS], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([NUM_TARGS], stddev=0.01))

    
