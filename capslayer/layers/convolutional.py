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

from capslayer.core import routing
from capslayer.core import transforming


def conv2d(inputs,
           activation,
           filters,
           out_caps_dims,
           kernel_size,
           strides,
           padding="valid",
           routing_method="EMRouting",
           name=None,
           reuse=None):
    """A 2D convolutional capsule layer.

    Args:
        inputs: A 6-D tensor with shape [batch_size, in_height, in_width, in_channels] + in_caps_dims.
        activation: A 4-D tensor with shape [batch_size, in_height, in_width, in_channels].
        filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
        out_caps_dims: A tuple/list of 2 integers, specifying the dimensions of output capsule, e.g. out_caps_dims=[4, 4] representing that each output capsule has shape [4, 4].
        kernel_size:  An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
        padding: One of "valid" or "same" (case-insensitive), now only support "valid".
        routing_method: One of "EMRouting" or "DynamicRouting", the method of routing-by-agreement algorithm.
        name: A string, the name of the layer.
        reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

    Returns:
        pose: A 6-D tensor with shape [batch_size, out_height, out_width, out_channesl] + out_caps_dims.
        activation: A 4-D tensor with shape [batch_size, out_height, out_width, out_channels].
    """

    name = "conv2d" if name is None else name
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        input_shape = cl.shape(inputs)
        input_rank = len(input_shape)
        activation_rank = len(activation.shape)
        if not input_rank == 6:
            raise ValueError('Inputs to `conv2d` should have rank 6. Received inputs rank:', str(input_rank))
        if not activation_rank == 4:
            raise ValueError('Activation to `conv2d` should have rank 4. Received activation rank:', str(activation_rank))

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size, input_shape[3]]
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
            kernel_size = [kernel_size[0], kernel_size[1], input_shape[3]]
        else:
            raise ValueError('"kernel_size" should be an integer or tuple/list of 2 integers. Received:', str(kernel_size))

        if isinstance(strides, int):
            strides = [strides, strides, 1]
        elif isinstance(strides, (list, tuple)) and len(strides) == 2:
            strides = [strides[0], strides[1], 1]
        else:
            raise ValueError('"strides" should be an integer or tuple/list of 2 integers. Received:', str(kernel_size))

        if not isinstance(out_caps_dims, (list, tuple)) or len(out_caps_dims) != 2:
            raise ValueError('"out_caps_dims" should be a tuple/list of 2 integers. Received:', str(out_caps_dims))
        elif isinstance(out_caps_dims, tuple):
            out_caps_dims = list(out_caps_dims)

        # 1. space to batch
        # patching everything into [batch_size, out_height, out_width, in_channels] + in_caps_dims (batched)
        # and [batch_size, out_height, out_width, in_channels] (activation).
        batched = cl.space_to_batch_nd(inputs, kernel_size, strides)
        activation = cl.space_to_batch_nd(activation, kernel_size, strides)

        # 2. transforming
        # transforming to [batch_size, out_height, out_width, in_channels, out_channels/filters] + out_caps_dims
        vote = transforming(batched,
                            num_outputs=filters,
                            out_caps_dims=out_caps_dims)

        # 3. routing
        pose, activation = routing(vote, activation, method=routing_method)

        return pose, activation


def conv3d(inputs,
           activation,
           filters,
           out_caps_dims,
           kernel_size,
           strides,
           padding="valid",
           routing_method="EMRouting",
           name=None,
           reuse=None):
    """A 3D convolutional capsule layer.

    Args:
        inputs: A 7-D tensor with shape [batch_size, in_depth, in_height, in_width, in_channels] + in_caps_dims.
        activation: A 5-D tensor with shape [batch_size, in_depth, in_height, in_width, in_channels].
        filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
        out_caps_dims: A tuple/list of 2 integers, specifying the dimensions of output capsule, e.g. out_caps_dims=[4, 4] representing that each output capsule has shape [4, 4].
        kernel_size:  An integer or tuple/list of 3 integers, specifying the height and width of the 3D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 3 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
        padding: One of "valid" or "same" (case-insensitive), now only support "valid".
        routing_method: One of "EMRouting" or "DynamicRouting", the method of routing-by-agreement algorithm.
        name: String, a name for the operation (optional).
        reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

    Returns:
        pose: A 7-D tensor with shape [batch_size, out_depth, out_height, out_width, out_channesl] + out_caps_dims.
        activation: A 5-D tensor with shape [batch_size, out_depth, out_height, out_width, out_channels].
    """

    name = "conv1d" if name is None else name
    with tf.name_scope(name):
        input_shape = cl.shape(inputs)
        input_rank = len(input_shape)
        activation_rank = len(activation.shape)
        if input_rank != 7:
            raise ValueError('Inputs to `conv3d` should have rank 7. Received input rank:', str(input_rank))
        if activation_rank != 5:
            raise ValueError('Activation to `conv3d` should have rank 5. Received input shape:', str(activation_rank))

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size, kernel_size]
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 3:
            kernel_size = kernel_size
        else:
            raise ValueError('"kernel_size" should be an integer or tuple/list of 3 integers. Received:', str(kernel_size))

        if isinstance(strides, int):
            strides = [strides, strides, strides]
        elif isinstance(strides, (list, tuple)) and len(strides) == 3:
            strides = strides
        else:
            raise ValueError('"strides" should be an integer or tuple/list of 3 integers. Received:', str(strides))

        if not isinstance(out_caps_dims, (list, tuple)) or len(out_caps_dims) != 2:
            raise ValueError('"out_caps_dims" should be a tuple/list of 2 integers. Received:', str(out_caps_dims))
        elif isinstance(out_caps_dims, tuple):
            out_caps_dims = list(out_caps_dims)

        # 1. space to batch
        batched = cl.space_to_batch_nd(inputs, kernel_size, strides)
        activation = cl.space_to_batch_nd(activation, kernel_size, strides)

        # 2. transforming
        vote = transforming(batched,
                            num_outputs=filters,
                            out_caps_dims=out_caps_dims)

        # 3. routing
        pose, activation = routing(vote, activation, method=routing_method)

        return pose, activation


def conv1d(inputs,
           activation,
           filters,
           out_caps_dims,
           kernel_size,
           stride,
           padding="valid",
           routing_method="EMRouting",
           name=None,
           reuse=None):
    """A 1D convolutional capsule layer (e.g. temporal convolution).

    Args:
        inputs: A 5-D tensor with shape [batch_size, in_width, in_channels] + in_caps_dims.
        activation: A 3-D tensor with shape [batch_size, in_width, in_channels].
        kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution.

    Returns:
        pose: A 5-D tensor with shape [batch_size, out_width, out_channesl] + out_caps_dims.
        activation: A 3-D tensor with shape [batch_size, out_width, out_channels].

    """

    name = "conv1d" if name is None else name
    with tf.variable_scope(name):
        input_shape = cl.shape(inputs)
        input_rank = len(input_shape)
        activation_rank = len(activation.shape)
        if input_rank != 5:
            raise ValueError('Inputs to `conv1d` should have rank 5. Received input rank:', str(input_rank))
        if activation_rank != 3:
            raise ValueError('Activation to `conv1d` should have rank 3. Received input shape:', str(activation_rank))

        if isinstance(kernel_size, int):
            kernel_size = [1, kernel_size]
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 1:
            kernel_size = [1, kernel_size[0]]
        else:
            raise ValueError('"kernel_size" should be an integer or tuple/list of 2 integers. Received:', str(kernel_size))

        if isinstance(stride, int):
            strides = [1, stride]
        elif isinstance(stride, (list, tuple)) and len(stride) == 1:
            strides = [1, stride[0]]
        else:
            raise ValueError('"stride" should be an integer or tuple/list of a single integer. Received:', str(stride))

        if not isinstance(out_caps_dims, (list, tuple)) or len(out_caps_dims) != 2:
            raise ValueError('"out_caps_dims" should be a tuple/list of 2 integers. Received:', str(out_caps_dims))
        elif isinstance(out_caps_dims, tuple):
            out_caps_dims = list(out_caps_dims)

        inputs = tf.expand_dims(inputs, axis=1)
        activation = tf.expand_dims(activation, axis=1)
        pose, activation = conv2d(inputs,
                                  activation,
                                  filters=filters,
                                  out_caps_dims=out_caps_dims,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  routing_method=routing_method,
                                  name="convolution",
                                  reuse=reuse)
        pose = tf.squeeze(pose, axis=1)
        activation = tf.squeeze(activation, axis=1)

    return pose, activation
