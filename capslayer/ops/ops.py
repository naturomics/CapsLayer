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


def squash(inputs, axis=-2, ord="euclidean", name=None):
    """Squashing function.

    Args:
        inputs: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1]
        ord: Order of the norm. Supported values are 'fro', 'euclidean', 1, 2, np.inf and any positive real number yielding the corresponding p-norm. Default is 'euclidean' which is equivalent to Frobenius norm if tensor is a matrix and equivalent to 2-norm for vectors.

    Returns:
        A tensor with the same shape as inputs but squashed in `axis` dimension.
    """

    name = "squashing" if name is None else name
    with tf.name_scope(name):
        norm = cl.norm(inputs, ord=ord, axis=axis, keepdims=True)
        norm_squared = tf.square(norm)
        scalar_factor = norm_squared / (1 + norm_squared)
        return scalar_factor * (inputs / norm)


def shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)
