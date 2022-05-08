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

import capslayer as cl
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf


def spread_loss(labels, logits, margin, regularizer=None):
    """
    Args:
        labels: [batch_size, num_label].
        logits: [batch_size, num_label].
        margin: Integer or 1-D Tensor.
        regularizer: use regularization.

    Returns:
        loss: Spread loss.
    """
    a_target = cl.reduce_sum(labels * logits, axis=1, keepdims=True)
    dist = (1 - labels) * margin - (a_target - logits)
    dist = tf.pow(tf.maximum(0., dist), 2)
    loss = tf.reduce_mean(tf.reduce_sum(dist, axis=-1))
    if regularizer is not None:
        regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += tf.reduce_mean(regularizer)
    return(loss)


def margin_loss(labels,
                logits,
                upper_margin=0.9,
                bottom_margin=0.1,
                downweight=0.5):
    """
    Args:
        labels: [batch_size, num_label].
        logits: [batch_size, num_label].
    """
    positive_selctor = tf.cast(tf.less(logits, upper_margin), tf.float32)
    positive_cost = positive_selctor * labels * tf.pow(logits - upper_margin, 2)

    negative_selctor = tf.cast(tf.greater(logits, bottom_margin), tf.float32)
    negative_cost = negative_selctor * (1 - labels) * tf.pow(logits - bottom_margin, 2)
    loss = 0.5 * positive_cost + 0.5 * downweight * negative_cost
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


def cross_entropy(labels, logits, regularizer=None):
    '''
    Args:
        ...

    Returns:
        ...
    '''
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    if regularizer is not None:
        regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += tf.reduce_mean(regularizer)
    return(loss)


def fcm_loss(inputs, probs, num_iters=5, m=2):
    """
    Args:
        inputs: [num_samples, 1, x_dims, 1]
        probs: [num_samples, num_clusters, 1, 1].
    """
    # centers: [1, num_clusters, x_dims]
    k = 1
    b = 0
    weights = []
    delta_probs = []
    for i in range(num_iters):
        probs_m = tf.pow(probs, m)
        centers = cl.reduce_sum(probs_m * inputs, axis=0, keepdims=True) / cl.reduce_sum(probs_m, axis=0, keepdims=True)
        # distance matrix with shape [num_samples, num_clusters, 1, 1]
        distance_matrix = cl.norm(inputs - centers, axis=(2, 3), keepdims=True)
        distance_matrix = tf.pow(distance_matrix, 2 / (m - 1))
        probs_plus = 1 / (distance_matrix / cl.reduce_sum(distance_matrix, axis=1, keepdims=True))
        delta_probs.append(tf.norm(probs_plus - probs))
        weights.append(tf.exp(tf.cast(k * i + b, tf.float32)))
        probs = probs_plus

    weights = tf.stack(weights, axis=0)
    delta_probs = tf.stack(delta_probs, axis=0)
    loss = tf.reduce_sum(weights * delta_probs) / tf.reduce_sum(weights)
    return loss
