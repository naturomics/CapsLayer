"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import tensorflow as tf

from config import cfg
from capslayer.utils import softmax
from capslayer.utils import get_batch_data
import capslayer


class CapsNet(object):
    def __init__(self, height=28, width=28, channels=1, num_label=10):
        '''
        Args:
            height: Integer, the height of input.
            width: Integer, the width of input.
            channels: Integer, the channels of input.
            num_label: Integer, the category number.
        '''
        self.height = height
        self.width =width
        self.channels = channels
        self.num_label = num_label

        self.graph = tf.Graph()
        with self.graph.as_default():
            if cfg.is_training:
                self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
                self.x = tf.reshape(self.X, shape=[cfg.batch_size, self.height, self.width, self.channels])
                self.Y = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                self.global_step = tf.Variable(1, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            else:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, None))
                self.x = tf.reshape(self.X, shape=[cfg.batch_size, self.height, self.width, self.channels])
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
                self.Y = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)
                self.build_arch()

            with tf.variable_scope('accuracy'):
                logits_idx = tf.to_int32(tf.argmax(softmax(self.activation, axis=1), axis=1))
                correct_prediction = tf.equal(tf.to_int32(self.labels), logits_idx)
                self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, return with shape [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.x, num_outputs=256, kernel_size=9, stride=1, padding='VALID')

        # return primaryCaps: [batch_size, 1152, 8, 1], activation: [batch_size, 1152]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps, activation = capslayer.layers.primaryCaps(conv1, filters=32, kernel_size=9, strides=2, out_caps_shape=[8, 1])

        # return digitCaps: [batch_size, num_label, 16, 1], activation: [batch_size, num_label]
        with tf.variable_scope('DigitCaps_layer'):
            primaryCaps = tf.reshape(primaryCaps, shape=[cfg.batch_size, -1, 8, 1])
            self.digitCaps, self.activation = capslayer.layers.fully_connected(primaryCaps, activation, num_outputs=10, out_caps_shape=[16, 1], routing_method='DynamicRouting')

        # Decoder structure in Fig. 2
        # Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            masked_caps = tf.multiply(self.digitCaps, tf.reshape(self.Y, (-1, self.num_label, 1, 1)))
            active_caps = tf.reshape(masked_caps, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(active_caps, num_outputs=512)
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=self.height * self.width * self.channels, activation_fn=tf.sigmoid)

    def loss(self):
        # 1. Margin loss

        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.activation))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.activation - cfg.m_minus))

        # reshape: [batch_size, num_label, 1, 1] => [batch_size, num_label]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, num_label]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, num_label], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.x, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.loss))
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, self.height, self.width, self.channels))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        train_summary.append(tf.summary.histogram('activation', self.activation))
        self.train_summary = tf.summary.merge(train_summary)
