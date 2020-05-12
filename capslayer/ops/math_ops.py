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


def matmul(a, b,
           transpose_a=False,
           transpose_b=False,
           name=None):
    name = "matmul" if name is None else name
    with tf.name_scope(name):
        if transpose_a:
            rank_a = len(a.shape)
            if rank_a < 2:
                raise TypeError("Rank must be greater than 2")
            perm = [i for i in range(rank_a - 2)]
            perm = perm + [rank_a - 1, rank_a - 2]
            a = tf.transpose(a, perm=perm)
        if transpose_b:
            rank_b = len(b.shape)
            if rank_b < 2:
                raise TypeError("Rank must be greater than 2")
            perm = [i for i in range(rank_b - 2)]
            perm = perm + [rank_b - 1, rank_b - 2]
            b = tf.transpose(b, perm=perm)
        a = tf.expand_dims(a, axis=-1)
        b = tf.expand_dims(b, axis=-3)
        c = tf.reduce_sum(a * b, axis=-2)
        return c


# Memory consumed!
def matmul_v1(a, b,
              transpose_a=False,
              transpose_b=False,
              name=None):
    name = "matmul" if name is None else name
    with tf.name_scope(name):
        rank_a = len(a.shape)
        rank_b = len(b.shape)
        if rank_a < 2 or rank_b < 2:
            raise TypeError("Rank must be greater than 2")
        perm_a = [i for i in range(rank_a - 2)]
        perm_b = [i for i in range(rank_b - 2)]

        if transpose_a:
            perm = [rank_a - 1] + perm_a + [rank_a - 2]
        else:
            perm = [rank_a - 2] + perm_a + [rank_a - 1]
        a = tf.transpose(a, perm=perm)

        if transpose_b:
            perm = [rank_b - 2] + perm_b + [rank_b - 1]
        else:
            perm = [rank_b - 1] + perm_b + [rank_b - 2]
        b = tf.transpose(b, perm=perm)

        C = []
        for i in range(a.get_shape()[0].value):
            B = []
            for j in range(b.get_shape()[0].value):
                k = tf.reduce_sum(a[i] * b[j], axis=-1, keepdims=True)
                B.append(k)
            C.append(tf.expand_dims(tf.concat(B, axis=-1), axis=-2))
        C = tf.concat(C, axis=-2)
        return(C)


def matmul_v2(a, b,
              transpose_a=False,
              transpose_b=False,
              name=None):
    name = "matmul" if name is None else name
    with tf.name_scope(name):
        rank_a = len(a.shape)
        rank_b = len(b.shape)
        if rank_a < 2 or rank_b < 2:
            raise TypeError("Rank must be greater than 2")
        if transpose_a:
            perm = [i for i in range(rank_a - 2)]
            perm = perm + [rank_a - 1, rank_a - 2]
            a = tf.transpose(a, perm=perm)
        if transpose_b:
            perm = [i for i in range(rank_b - 2)]
            perm = perm + [rank_b - 1, rank_b - 2]
            b = tf.transpose(b, perm=perm)

        b = tf.tile(b, [1 for i in range(rank_b - 2)] + [cl.shape(a)[-2], 1])
        shape = cl.shape(a)[:-2] + [np.prod(cl.shape(a)[-2:]), 1]
        a_prime = tf.reshape(a, shape=shape)
        c = a_prime * b
        shape = cl.shape(a) + cl.shape(b)[-1:]
        c = tf.reshape(c, shape=shape)
        c = tf.reduce_sum(c, axis=-2)

    return c


def norm(tensor, ord='euclidean', axis=None, keepdims=None, name=None):
    try:
        return tf.norm(tensor, ord=ord, axis=axis, keepdims=keepdims, name=name)
    except:
        return tf.norm(tensor, ord=ord, axis=axis, keep_dims=keepdims, name=name)


def reduce_sum(input_tensor,
               axis=None,
               keepdims=None,
               name=None):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims, name=name)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims, name=name)


def divide(x, y, safe_mode=True, epsilon=None, name=None):
    """ A wrapper of `tf.divide`, computes Python style division of x by y but extends safe divide support.
        If safe_mode is `True` or epsilon is given(a small float number), the absolute value of denominator
        in the division will be clip to make sure it's bigger than epsilon(default is 1e-13).

    Args:
        safe_mode: Use safe divide mode.
        epsilon: Float number. Default is `1e-13`.
    """
    if not safe_mode and epsilon is None:
        return tf.divide(x, y, name=name)
    else:
        epsilon = 1e-20 if epsilon is None else epsilon
        name = "safe_divide" if name is None else name
        with tf.name_scope(name):
            y = tf.where(tf.greater(tf.abs(y), epsilon), y, y + tf.sign(y) * epsilon)
            return tf.divide(x, y)


def log(x, epsilon=1e-20, name=None):
    """ A wrapper of `tf.log`, computes natural logarithm of x element-wise but extends safe log support.
        If epsilon is given as a positive float, x will be clipped to bigger than epsilon before doing computing.
    """
    if isinstance(epsilon, float) and epsilon > 0:
        return tf.log(tf.maximum(x, epsilon), name=name)
    else:
        return tf.log(x, name=name)


if __name__ == "__main__":
    a = tf.constant([[[1, 2], [3, 4], [5, 6]]], dtype=tf.float32)
    b = tf.constant([[[7, 8, 9], [10, 11, 12], [13, 14, 15]]],
                    dtype=tf.float32)
    c1 = tf.matmul(a, b, transpose_a=True, transpose_b=True)
    c2 = matmul_v2(a, b, transpose_a=True, transpose_b=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print(sess.run(c1))
    print(sess.run(c2))
