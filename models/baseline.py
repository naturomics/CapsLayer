# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import capslayer as cl
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

from config import cfg


class Model(object):
    def __init__(self, height, width, channels, num_label):
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

    def create_network(self, inputs, labels):
        self.labels = labels
        self.y = tf.one_hot(labels, depth=self.num_label, axis=-1, dtype=tf.float32)
        inputs = tf.reshape(inputs, shape=[-1, self.height, self.width, self.channels])
        conv1 = tf.layers.conv2d(inputs, filters=256, kernel_size=5, activation=tf.nn.relu)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")

        conv2 = tf.layers.conv2d(pool1, filters=256, kernel_size=5, activation=tf.nn.relu)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")

        conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=5, activation=tf.nn.relu)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")

        input = tf.reshape(pool3, shape=(-1, np.prod(pool3.get_shape()[1:])))
        fc1 = tf.layers.dense(input, units=328)
        fc2 = tf.layers.dense(fc1, units=192)
        out = tf.layers.dense(fc2, units=self.num_label, activation=None)
        self.y_pred = out
        self.probs = tf.nn.softmax(self.y_pred, axis=1)

        with tf.variable_scope('accuracy'):
            logits_idx = tf.to_int32(tf.argmax(self.probs, axis=1))
            correct_prediction = tf.equal(tf.to_int32(self.labels), logits_idx)
            correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(correct / tf.cast(tf.shape(self.probs)[0], tf.float32))
            cl.summary.scalar('accuracy', self.accuracy, verbose=cfg.summary_verbose)

    def _loss(self):
        return tf.losses.softmax_cross_entropy(self.y, self.y_pred)

    def train(self, optimizer, num_gpus=1):
        self.global_step = tf.Variable(1, name='global_step', trainable=False)
        total_loss = self._loss()
        optimizer = tf.train.AdamOptimizer()
        train_ops = optimizer.minimize(total_loss, global_step=self.global_step)
        summary_ops = tf.summary.merge_all()

        return(total_loss, train_ops, summary_ops)
