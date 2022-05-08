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

"""
This module provides a set of high-level capsule networks layers.
"""

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

from capslayer.core import routing
from capslayer.core import transforming


def dense(inputs, activation,
          num_outputs,
          out_caps_dims,
          routing_method='EMRouting',
          coordinate_addition=False,
          reuse=None,
          name=None):
    """A fully connected capsule layer.

    Args:
        inputs: A 4-D tensor with shape [batch_size, num_inputs] + in_caps_dims or [batch_size, in_height, in_width, in_channels] + in_caps_dims
        activation: [batch_size, num_inputs] or [batch_size, in_height, in_width, in_channels]
        num_outputs: Integer, the number of output capsules in the layer.
        out_caps_dims: A list with two elements, pose shape of output capsules.

    Returns:
        pose: A 4-D tensor with shape [batch_size, num_outputs] + out_caps_dims
        activation: [batch_size, num_outputs]
    """
    name = "dense" if name is None else name
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse()
        if coordinate_addition and len(inputs.shape) == 6 and len(activation.shape) == 4:
            vote = transforming(inputs, num_outputs=num_outputs, out_caps_dims=out_caps_dims)
            with tf.name_scope("coodinate_addition"):
                batch_size, in_height, in_width, in_channels, _, out_caps_height, out_caps_width = cl.shape(vote)
                num_inputs = in_height * in_width * in_channels

                zeros = np.zeros((in_height, out_caps_width - 1))
                coord_offset_h = ((np.arange(in_height) + 0.5) / in_height).reshape([in_height, 1])
                coord_offset_h = np.concatenate([zeros, coord_offset_h], axis=-1)
                zeros = np.zeros((out_caps_height - 1, out_caps_width))
                coord_offset_h = np.stack([np.concatenate([coord_offset_h[i:(i + 1), :], zeros], axis=0) for i in range(in_height)], axis=0)
                coord_offset_h = coord_offset_h.reshape((1, in_height, 1, 1, 1, out_caps_height, out_caps_width))

                zeros = np.zeros((1, in_width))
                coord_offset_w = ((np.arange(in_width) + 0.5) / in_width).reshape([1, in_width])
                coord_offset_w = np.concatenate([zeros, coord_offset_w, zeros, zeros], axis=0)
                zeros = np.zeros((out_caps_height, out_caps_width - 1))
                coord_offset_w = np.stack([np.concatenate([zeros, coord_offset_w[:, i:(i + 1)]], axis=1) for i in range(in_width)], axis=0)
                coord_offset_w = coord_offset_w.reshape((1, 1, in_width, 1, 1, out_caps_height, out_caps_width))

                vote = vote + tf.constant(coord_offset_h + coord_offset_w, dtype=tf.float32)

                vote = tf.reshape(vote, shape=[batch_size, num_inputs, num_outputs] + out_caps_dims)
                activation = tf.reshape(activation, shape=[batch_size, num_inputs])

        elif len(inputs.shape) == 4 and len(activation.shape) == 2:
            vote = transforming(inputs, num_outputs=num_outputs, out_caps_dims=out_caps_dims)

        else:
            raise TypeError("Wrong rank for inputs or activation")

        pose, activation = routing(vote, activation, routing_method)
        # pose, activation = cl.core.gluing(vote, activation)
        assert len(pose.shape) == 4
        assert len(activation.shape) == 2

    return(pose, activation)


def primaryCaps(inputs, filters,
                kernel_size,
                strides,
                out_caps_dims,
                method=None,
                name=None):
    '''Primary capsule layer.

    Args:
        inputs: [batch_size, in_height, in_width, in_channels].
        filters: Integer, the dimensionality of the output space.
        kernel_size: kernel_size
        strides: strides
        out_caps_dims: A list of 2 integers.
        method: the method of calculating probability of entity existence(logistic, norm, None)

    Returns:
        pose: A 6-D tensor, [batch_size, out_height, out_width, filters] + out_caps_dims
        activation: A 4-D tensor, [batch_size, out_height, out_width, filters]
    '''

    name = "primary_capsule" if name is None else name
    with tf.variable_scope(name):
        channels = filters * np.prod(out_caps_dims)
        channels = channels + filters if method == "logistic" else channels
        pose = tf.layers.conv2d(inputs, channels,
                                kernel_size=kernel_size,
                                strides=strides, activation=None)
        shape = cl.shape(pose, name="get_pose_shape")
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        shape = [batch_size, height, width, filters] + out_caps_dims

        if method == 'logistic':
            # logistic activation unit
            pose, activation_logit = tf.split(pose, [channels - filters, filters], axis=-1)
            pose = tf.reshape(pose, shape=shape)
            activation = tf.sigmoid(activation_logit)
        elif method == 'norm' or method is None:
            pose = tf.reshape(pose, shape=shape)
            squash_on = -2 if out_caps_dims[-1] == 1 else [-2, -1]
            pose = cl.ops.squash(pose, axis=squash_on)
            activation = cl.norm(pose, axis=(-2, -1))
        activation = tf.clip_by_value(activation, 1e-20, 1. - 1e-20)

        return(pose, activation)
