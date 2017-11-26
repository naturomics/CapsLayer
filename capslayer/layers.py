'''
This module provides a set of high-level neural networks layers.
'''

import tensorflow as tf
from functools import reduce

from capslayer.utils import get_transformation_matrix_shape
from capslayer.utils import euclidean_norm
from capslayer.ops import routing


def fully_connected(inputs, activation,
                    num_outputs,
                    out_caps_shape,
                    routing_method='EMRouting',
                    reuse=None):
    '''A capsule fully connected layer.
    Args:
        inputs: A tensor with shape [batch_size, num_inputs] + in_caps_shape.
        activation: [batch_size, num_inputs]
        num_outputs: Integer, the number of output capsules in the layer.
        out_caps_shape: A list with two elements, pose shape of output capsules.
    Returns:
        pose: [batch_size, num_outputs] + out_caps_shape
        activation: [batch_size, num_outputs]
    '''
    in_pose_shape = inputs.get_shape().as_list()
    num_inputs = in_pose_shape[1]
    batch_size = in_pose_shape[0]
    T_size = get_transformation_matrix_shape(in_pose_shape[-2:], out_caps_shape)
    T_shape = [1, num_inputs, num_outputs] + T_size
    T_matrix = tf.get_variable("transformation_matrix", shape=T_shape)
    T_matrix = tf.tile(T_matrix, [batch_size, 1, 1, 1, 1])
    inputs = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, num_outputs, 1, 1])
    with tf.variable_scope('transformation'):
        # vote: [batch_size, num_inputs, num_outputs] + out_caps_shape
        vote = tf.matmul(T_matrix, inputs)
    with tf.variable_scope('routing'):
        if routing_method == 'EMRouting':
            activation = tf.reshape(activation, shape=activation.get_shape().as_list() + [1, 1])
            vote = tf.reshape(vote, shape=[batch_size, num_inputs, num_outputs, -1])
            pose, activation = routing(vote, activation, num_outputs, out_caps_shape, routing_method)
            pose = tf.reshape(pose, shape=[batch_size, num_outputs] + out_caps_shape)
            activation = tf.reshape(activation, shape=[batch_size, -1])
        elif routing_method == 'DynamicRouting':
            pose, _ = routing(vote, activation, num_outputs=num_outputs, out_caps_shape=out_caps_shape, method=routing_method)
            pose = tf.squeeze(pose, axis=1)
            activation = tf.squeeze(euclidean_norm(pose))
    return(pose, activation)


def primaryCaps(input, filters,
                kernel_size,
                strides,
                out_caps_shape,
                method=None,
                regularizer=None):
    '''PrimaryCaps layer
    Args:
        input: [batch_size, in_height, in_width, in_channels].
        filters: Integer, the dimensionality of the output space.
        kernel_size: ...
        strides: ...
        out_caps_shape: ...
        method: the method of calculating probability of entity existence(logistic, norm, None)
    Returns:
        pose: [batch_size, out_height, out_width, filters] + out_caps_shape
        activation: [batch_size, out_height, out_width, filters]
    '''
    # pose matrix
    pose_size = reduce(lambda x, y: x * y, out_caps_shape)
    pose = tf.layers.conv2d(input, filters * pose_size,
                            kernel_size=kernel_size,
                            strides=strides, activation=None,
                            activity_regularizer=regularizer)
    pose_shape = pose.get_shape().as_list()[:3] + [filters] + out_caps_shape
    pose = tf.reshape(pose, shape=pose_shape)

    if method == 'logistic':
        # logistic activation unit
        activation = tf.layers.conv2d(input, filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      activation=tf.nn.sigmoid,
                                      activity_regularizer=regularizer)
    elif method == 'norm':
        activation = euclidean_norm(pose)
    else:
        activation = None

    return(pose, activation)


def conv2d(in_pose,
           activation,
           filters,
           out_caps_shape,
           kernel_size,
           strides=(1, 1),
           coordinate_addition=False,
           regularizer=None,
           reuse=None):
    '''A capsule convolutional layer.
    Args:
        in_pose: A tensor with shape [batch_size, in_height, in_width, in_channels] + in_caps_shape.
        activation: A tensor with shape [batch_size, in_height, in_width, in_channels]
        filters: ...
        out_caps_shape: ...
        kernel_size: ...
        strides: ...
        coordinate_addition: ...
        regularizer: apply regularization on a newly created variable and add the variable to the collection tf.GraphKeys.REGULARIZATION_LOSSES.
        reuse: ...
    Returns:
        out_pose: A tensor with shape [batch_size, out_height, out_height, out_channals] + out_caps_shape,
        out_activation: A tensor with shape [batch_size, out_height, out_height, out_channels]
    '''
    # do some preparation stuff
    in_pose_shape = in_pose.get_shape().as_list()
    in_caps_shape = in_pose_shape[-2:]
    batch_size = in_pose_shape[0]
    in_channels = in_pose_shape[3]

    T_size = get_transformation_matrix_shape(in_caps_shape, out_caps_shape)
    if isinstance(kernel_size, int):
        h_kernel_size = kernel_size
        w_kernel_size = kernel_size
    elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
        h_kernel_size = kernel_size[0]
        w_kernel_size = kernel_size[1]
    if isinstance(strides, int):
        h_stride = strides
        w_stride = strides
    elif isinstance(strides, (list, tuple)) and len(strides) == 2:
        h_stride = strides[0]
        w_stride = strides[1]
    num_inputs = h_kernel_size * w_kernel_size * in_channels
    batch_shape = [batch_size, h_kernel_size, w_kernel_size, in_channels]
    T_shape = (1, num_inputs, filters) + tuple(T_size)

    T_matrix = tf.get_variable("transformation_matrix", shape=T_shape, regularizer=regularizer)
    T_matrix_batched = tf.tile(T_matrix, [batch_size, 1, 1, 1, 1])

    h_step = int((in_pose_shape[1] - h_kernel_size) / h_stride + 1)
    w_step = int((in_pose_shape[2] - w_kernel_size) / w_stride + 1)
    out_pose = []
    out_activation = []
    # start to do capsule convolution.
    # Note: there should be another way more computationally efficient to do this
    for i in range(h_step):
        col_pose = []
        col_prob = []
        h_s = i * h_stride
        h_e = h_s + h_kernel_size
        for j in range(w_step):
            with tf.variable_scope("transformation"):
                begin = [0, i * h_stride, j * w_stride, 0, 0, 0]
                size = batch_shape + in_caps_shape
                w_s = j * w_stride
                pose_sliced = in_pose[:, h_s:h_e, w_s:(w_s + w_kernel_size), :, :, :]
                pose_reshaped = tf.reshape(pose_sliced, shape=[batch_size, num_inputs, 1] + in_caps_shape)
                shape = [batch_size, num_inputs, filters] + in_caps_shape
                batch_pose = tf.multiply(pose_reshaped, tf.constant(1., shape=shape))
                vote = tf.reshape(tf.matmul(T_matrix_batched, batch_pose), shape=[batch_size, num_inputs, filters, -1])
                # do Coordinate Addition. Note: not yet completed
                if coordinate_addition:
                    x = j / w_step
                    y = i / h_step

            with tf.variable_scope("routing") as scope:
                if i > 0 or j > 0:
                    scope.reuse_variables()
                begin = [0, i * h_stride, j * w_stride, 0]
                size = [batch_size, h_kernel_size, w_kernel_size, in_channels]
                prob = tf.slice(activation, begin, size)
                prob = tf.reshape(prob, shape=[batch_size, -1, 1, 1])
                pose, prob = routing(vote, prob, filters, out_caps_shape, method="EMRouting", regularizer=regularizer)
            col_pose.append(pose)
            col_prob.append(prob)
        col_pose = tf.concat(col_pose, axis=2)
        col_prob = tf.concat(col_prob, axis=2)
        out_pose.append(col_pose)
        out_activation.append(col_prob)
    out_pose = tf.concat(out_pose, axis=1)
    out_activation = tf.concat(out_activation, axis=1)

    return(out_pose, out_activation)
