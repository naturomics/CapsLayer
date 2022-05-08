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


def transforming(inputs, num_outputs, out_caps_dims, name=None):
    """
    Args:
        inputs: A 4-D or 6-D tensor, [batch_size, num_inputs] + in_caps_dims or [batch_size, height, width, channels] + in_caps_dims.
        num_outputs: Integer, the number of output capsules.
        out_caps_dims: A list of 2 integers. The dimensions of output capsule, e.g. out_caps_dims=[4, 4].
        name: String, a name for this operation.

    Returns:
        votes: A 5-D or 7-D tensor, [batch_size, num_inputs, num_outputs] + out_caps_dims or [batch_size, height, width, channels, num_outputs] + out_caps_dims.
    """
    name = "transforming" if name is None else name
    with tf.variable_scope(name) as scope:
        input_shape = cl.shape(inputs)
        prefix_shape = [1 for i in range(len(input_shape) - 3)] + input_shape[-3:-2] + [num_outputs]
        in_caps_dims = input_shape[-2:]
        if in_caps_dims[0] == out_caps_dims[1]:
            shape = prefix_shape + [out_caps_dims[0], 1, in_caps_dims[1]]
            expand_axis = -3
            reduce_sum_axis = -1
        elif in_caps_dims[1] == out_caps_dims[0]:
            shape = prefix_shape + [in_caps_dims[0], 1, out_caps_dims[1]]
            expand_axis = -1
            reduce_sum_axis = -3
        elif in_caps_dims[0] == out_caps_dims[0]:
            shape = prefix_shape + [1, out_caps_dims[1], in_caps_dims[1]]
            expand_axis = -2
            reduce_sum_axis = -1
        elif in_caps_dims[1] == out_caps_dims[1]:
            shape = prefix_shape + [in_caps_dims[0], out_caps_dims[0], 1]
            expand_axis = -2
            reduce_sum_axis = -3
        else:
            raise TypeError("out_caps_dims must have at least one value being the same with the in_caps_dims")
        in_pose = tf.expand_dims(inputs, axis=-3)
        ones = tf.ones(shape=prefix_shape + [1, 1])
        in_pose = tf.expand_dims(in_pose * ones, axis=expand_axis)
        transform_mat = tf.get_variable("transformation_matrix", shape=shape)
        votes = tf.reduce_sum(in_pose * transform_mat, axis=reduce_sum_axis)

        return votes
