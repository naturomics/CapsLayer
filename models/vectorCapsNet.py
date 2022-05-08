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


class CapsNet(object):
    def __init__(self, height=28, width=28, channels=1, num_label=10):
        '''

        Args:
            height: Integer, the height of inputs.
            width: Integer, the width of inputs.
            channels: Integer, the channels of inputs.
            num_label: Integer, the category number.
        '''
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

    def create_network(self, inputs, labels):
        """ Setup capsule network.

        Args:
            inputs: Tensor or array with shape [batch_size, height, width, channels] or [batch_size, height * width * channels].
            labels: Tensor or array with shape [batch_size].

        Returns:
            poses: [batch_size, num_label, 16, 1].
            probs: Tensor with shape [batch_size, num_label], the probability of entity presence.
        """
        self.raw_imgs = inputs
        self.labels = labels
        with tf.variable_scope('Conv1_layer'):
            # Conv1, return with shape [batch_size, 20, 20, 256]
            inputs = tf.reshape(self.raw_imgs, shape=[-1, self.height, self.width, self.channels])
            conv1 = tf.layers.conv2d(inputs,
                                     filters=256,
                                     kernel_size=9,
                                     strides=1,
                                     padding='VALID',
                                     activation=tf.nn.relu)

        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps, activation = cl.layers.primaryCaps(conv1,
                                                            filters=32,
                                                            kernel_size=9,
                                                            strides=2,
                                                            out_caps_dims=[8, 1],
                                                            method="norm")

        with tf.variable_scope('DigitCaps_layer'):
            routing_method = "EMRouting"
            num_inputs = np.prod(cl.shape(primaryCaps)[1:4])
            primaryCaps = tf.reshape(primaryCaps, shape=[-1, num_inputs, 8, 1])
            activation = tf.reshape(activation, shape=[-1, num_inputs])
            self.poses, self.probs = cl.layers.dense(primaryCaps,
                                                     activation,
                                                     num_outputs=self.num_label,
                                                     out_caps_dims=[16, 1],
                                                     routing_method=routing_method)
            cl.summary.histogram('activation', self.probs, verbose=cfg.summary_verbose)

        # Decoder structure
        # Reconstructe the inputs with 3 FC layers
        with tf.variable_scope('Decoder'):
            labels = tf.one_hot(self.labels, depth=self.num_label, axis=-1, dtype=tf.float32)
            self.labels_one_hoted = tf.reshape(labels, (-1, self.num_label, 1, 1))
            masked_caps = tf.multiply(self.poses, self.labels_one_hoted)
            num_inputs = np.prod(masked_caps.get_shape().as_list()[1:])
            active_caps = tf.reshape(masked_caps, shape=(-1, num_inputs))
            fc1 = tf.layers.dense(active_caps, units=512, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=1024, activation=tf.nn.relu)
            num_outputs = self.height * self.width * self.channels
            self.recon_imgs = tf.layers.dense(fc2,
                                              units=num_outputs,
                                              activation=tf.sigmoid)
            recon_imgs = tf.reshape(self.recon_imgs, shape=[-1, self.height, self.width, self.channels])
            cl.summary.image('reconstruction_img', recon_imgs, verbose=cfg.summary_verbose)

        with tf.variable_scope('accuracy'):
            logits_idx = tf.to_int32(tf.argmax(cl.softmax(self.probs, axis=1), axis=1))
            correct_prediction = tf.equal(tf.to_int32(self.labels), logits_idx)
            correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(correct / tf.cast(tf.shape(self.probs)[0], tf.float32))
            cl.summary.scalar('accuracy', self.accuracy, verbose=cfg.summary_verbose)

        return self.poses, self.probs

    def _loss(self):
        with tf.variable_scope("loss"):
            # 1. Margin loss
            margin_loss = cl.losses.margin_loss(logits=self.probs,
                                                labels=tf.squeeze(self.labels_one_hoted, axis=(2, 3)))
            cl.summary.scalar('margin_loss', margin_loss, verbose=cfg.summary_verbose)

            # 2. The reconstruction loss
            orgin = tf.reshape(self.raw_imgs, shape=(-1, self.height * self.width * self.channels))
            squared = tf.square(self.recon_imgs - orgin)
            reconstruction_err = tf.reduce_mean(squared)
            cl.summary.scalar('reconstruction_loss', reconstruction_err, verbose=cfg.summary_verbose)

            # 3. Total loss
            # The paper uses sum of squared error as reconstruction error, but we
            # have used reduce_mean in `# 2 The reconstruction loss` to calculate
            # mean squared error. In order to keep in line with the paper,the
            # regularization scale should be 0.0005*784=0.392
            total_loss = margin_loss + cfg.regularization_scale * reconstruction_err

            cl.summary.scalar('total_loss', total_loss, verbose=cfg.summary_verbose)
            return total_loss

    def train(self, optimizer, num_gpus=1):
        self.global_step = tf.Variable(1, name='global_step', trainable=False)
        total_loss = self._loss()
        optimizer = tf.train.AdamOptimizer()
        train_ops = optimizer.minimize(total_loss, global_step=self.global_step)
        summary_ops = tf.summary.merge_all()

        return(total_loss, train_ops, summary_ops)
