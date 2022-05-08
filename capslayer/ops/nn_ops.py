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

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import capslayer as cl
import numpy as np


def space_to_batch_nd(input, kernel_size, strides, name=None):
    """ Space to batch with strides. Different to tf.space_to_batch_nd.
        for convCapsNet model: memory 4729M, speed 0.165 sec/step, similiar to space_to_batch_nd_v1

    Args:
        input: A Tensor. N-D with shape input_shape = [batch] + spatial_shape + remaining_shape, where spatial_shape has M dimensions.
        kernel_size: A sequence of len(spatial_shape)-D positive integers specifying the spatial dimensions of the filters.
        strides: A sequence of len(spatial_shape)-D positive integers specifying the stride at which to compute output.

    Returns:
        A Tensor.
    """
    assert len(kernel_size) == 3
    assert len(strides) == 3
    name = "space_to_batch_nd" if name is None else name
    with tf.name_scope(name):
        input_shape = cl.shape(input)
        h_steps = int((input_shape[1] - kernel_size[0]) / strides[0] + 1)
        w_steps = int((input_shape[2] - kernel_size[1]) / strides[1] + 1)
        d_steps = int((input_shape[3] - kernel_size[2]) / strides[2] + 1)
        blocks = []  # each element with shape [batch, h_kernel_size * w_kernel_size * d_kernel_size] + remaining_shape
        for d in range(d_steps):
            d_s = d * strides[2]
            d_e = d_s + kernel_size[2]
            h_blocks = []
            for h in range(h_steps):
                h_s = h * strides[0]
                h_e = h_s + kernel_size[0]
                w_blocks = []
                for w in range(w_steps):
                    w_s = w * strides[1]
                    w_e = w_s + kernel_size[1]
                    block = input[:, h_s:h_e, w_s:w_e, d_s:d_e]
                    # block = tf.reshape(block, shape=[tf.shape(input)[0], np.prod(kernel_size)] + input_shape[4:])
                    w_blocks.append(block)
                h_blocks.append(tf.concat(w_blocks, axis=2))
            blocks.append(tf.concat(h_blocks, axis=1))
        return tf.concat(blocks, axis=0)


def space_to_batch_nd_v1(inputs, kernel_size, strides, name=None):
    """ for convCapsNet model: memory 4719M, speed 0.169 sec/step
    """
    name = "space_to_batch_nd" if name is None else name
    with tf.name_scope(name):
        height, width, depth = cl.shape(inputs)[1:4]
        h_offsets = [[(h + k) for k in range(0, kernel_size[0])] for h in range(0, height + 1 - kernel_size[0], strides[0])]
        w_offsets = [[(w + k) for k in range(0, kernel_size[1])] for w in range(0, width + 1 - kernel_size[1], strides[1])]
        d_offsets = [[(d + k) for k in range(0, kernel_size[2])] for d in range(0, depth + 1 - kernel_size[2], strides[2])]
        patched = tf.gather(inputs, h_offsets, axis=1)
        patched = tf.gather(patched, w_offsets, axis=3)
        patched = tf.gather(patched, d_offsets, axis=5)

        if len(patched.shape) == 7:
            perm = [0, 1, 3, 5, 2, 4, 6]
        else:
            perm = [0, 1, 3, 5, 2, 4, 6, 7, 8]

        patched = tf.transpose(patched, perm=perm)
        shape = cl.shape(patched)

        if depth == kernel_size[2]:   # for conv2d
            shape = shape[:3] + [np.prod(shape[3:-2])] + shape[-2:] if len(patched.shape) == 9 else shape[:3] + [np.prod(shape[3:])]
        else:                         # for conv3d
            shape = shape[:4] + [np.prod(shape[4:-2])] + shape[-2:] if len(patched.shape) == 9 else shape[:4] + [np.prod(shape[4:])]

        patched = tf.reshape(patched, shape=shape)
    return patched


def batch_to_space_nd(input, spatial_shape, name=None):
    name = "batch_to_space_nd" if name is None else name
    with tf.name_scope(name):
        input_shape = cl.shape(input)
        shape = [-1] + spatial_shape + input_shape[1:]
        return tf.reshape(input, shape=shape)


def softmax(logits, axis=None, name=None):
    name = "Softmax" if name is None else name
    with tf.name_scope(name):
        if axis < 0:
            axis = len(logits.shape) + axis
        try:
            return tf.nn.softmax(logits, axis=axis)
        except:
            return tf.nn.softmax(logits, dim=axis)
