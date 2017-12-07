import tensorflow as tf

import capslayer
from config import cfg
from capslayer.utils import get_batch_data


class Model(object):
    def __init__(self, height, width, channels, num_label):
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

        self.graph = tf.Graph()
        with self.graph.as_default():
            # self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, self.height, self.width, self.channels))
            # self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
            self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
            self.x = tf.reshape(self.X, shape=[cfg.batch_size, self.height, self.width, self.channels])
            self.y = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)

            self.setup_model()
            self.loss()
            self._summary()

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            correct_prediction = tf.equal(tf.to_int32(tf.argmax(self.y_pred, axis=1)), self.labels)
            self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    def setup_model(self):
        conv1 = tf.layers.conv2d(self.x, filters=256, kernel_size=5, activation=tf.nn.relu)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")

        conv2 = tf.layers.conv2d(pool1, filters=256, kernel_size=5, activation=tf.nn.relu)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")

        conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=5, activation=tf.nn.relu)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")

        input = tf.reshape(pool3, shape=(cfg.batch_size, -1))
        fc1 = tf.contrib.layers.fully_connected(input, num_outputs=328)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=192)
        out = tf.contrib.layers.fully_connected(fc2, num_outputs=10, activation_fn=None)
        self.y_pred = out
        self.activation = tf.nn.softmax(self.y_pred)

    def loss(self):
        self.loss = tf.losses.softmax_cross_entropy(self.y, self.y_pred)

    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/loss', self.loss))
        self.train_summary = tf.summary.merge(train_summary)
