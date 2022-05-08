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


def routing(votes,
            activation=None,
            method='EMRouting',
            num_iter=3,
            name=None):
    """ Routing-by-agreement algorithm.

    Args:
        votes: A 5-D or 7-D tensor, the transformed matrices of the layer below, which shape should be [batch_size, ..., in_channels/num_inputs, num_outputs] + out_caps_dims and the routing will be performed on the last four dimensions.
        activation: A 2-D or 4-D tensor, [batch_size, num_inputs] or [batch_size, out_height, out_width, in_channels].
        method: One of 'EMRouting' or 'DynamicRouting', the method for updating coupling coefficients between votes and pose.
        num_iter: Integer, the number of routing iterations.
        name: String, a name for the operation.

    Returns:
        pose: A 4-D or 6-D tensor, [batch_size, num_outputs] + out_caps_dims or [batch_size, out_height, out_width, out_channels] + out_caps_dims.
        activation: A 2-D or 4-D tensor, [batch_size, num_outputs] or [batch_size, out_height, out_width, out_channels].
    """
    vote_rank = len(votes.shape)
    activation_rank = len(activation.shape)
    if vote_rank != 5 and vote_rank != 7:
        raise ValueError('Votes to "routing" should have rank 5 or 7 but it is rank', str(vote_rank))
    if activation_rank != 2 and activation_rank != 4:
        raise ValueError('Activation to "routing" should have rank 2 or 4 but it is rank', str(activation_rank))

    name = "routing" if name is None else name
    with tf.variable_scope(name):
        if method == 'EMRouting':
            pose, activation = EMRouting(votes, activation, num_iter)
        elif method == 'DynamicRouting':
            if not votes.shape[-1].value == 1:
                raise ValueError('"DynamicRouting" method only works for vetocr, please set the "out_caps_dims" like [n, 1]')
            pose, activation = dynamicRouting(votes, num_iter, leaky=True)
            # pose, activation = dynamicRouting_v1(votes, num_iter, leaky=True)
        else:
            raise Exception('Invalid routing method!', method)

    activation = tf.clip_by_value(activation, 1e-30, 1. - 1e-30)
    assert len(pose.shape) == 4 or len(pose.shape) == 6
    assert len(activation.shape) == 2 or len(activation.shape) == 4
    return(pose, activation)


def dynamicRouting(votes,
                   num_routing=3,
                   use_bias=True,
                   leaky=True):
    """ Dynamic routing algorithm.

    See [Sabour et al., 2017](https://arxiv.org/abs/1710.09829).

    Args:
        votes: A 5-D or 7-D tensor with shape [batch_size, ..., in_channels/num_inputs, num_outputs] + out_caps_dims.
        num_routing: Integer, number of routing iterations.
        use_bias: Boolean, whether the layer uses a bias.
        leaky: Boolean, whether the algorithm uses leaky routing.

    Returns:
        poses: A 4-D or 6-D tensor.
        probs: A 2-D or 4-D tensor.
    """

    vote_shape = cl.shape(votes)
    logit_shape = vote_shape[:-2] + [1, 1]
    logits = tf.fill(logit_shape, 0.0)
    squash_on = -2 if vote_shape[-1] == 1 else [-2, -1]
    if use_bias:
        bias_shape = [1 for i in range(len(vote_shape) - 3)] + vote_shape[-3:]
        biases = tf.get_variable("biases",
                                 bias_shape,
                                 initializer=tf.constant_initializer(0.1),
                                 dtype=tf.float32)

    def _body(i, logits, poses):
        if leaky:
            route = _leaky_routing(logits)
        else:
            route = tf.nn.softmax(logits, axis=-3)

        if use_bias:
            preactivate = cl.reduce_sum(route * votes, axis=-4, keepdims=True) + biases
        else:
            preactivate = cl.reduce_sum(route * votes, axis=-4, keepdims=True)

        pose = cl.ops.squash(preactivate, axis=squash_on)
        poses = poses.write(i, pose)

        if vote_shape[-1] == 1:
            distances = cl.matmul(votes, pose, transpose_a=True)
        else:
            diff = votes - pose
            distances = tf.trace(cl.matmul(diff, diff))[..., tf.newaxis, tf.newaxis]
        logits += distances

        return(i + 1, logits, poses)

    poses = tf.TensorArray(dtype=tf.float32,
                           size=num_routing,
                           clear_after_read=False)
    i = tf.constant(0, dtype=tf.int32)
    _, logits, poses = tf.while_loop(lambda i, logits, poses: i < num_routing,
                                     _body,
                                     loop_vars=[i, logits, poses],
                                     swap_memory=True)
    poses = tf.squeeze(poses.read(num_routing - 1), axis=-4)
    probs = tf.norm(poses, axis=[-2, -1])

    return poses, probs


def _leaky_routing(logits):
    leak_shape = cl.shape(logits)
    leak = tf.zeros(leak_shape[:-3] + [1, 1, 1])
    leaky_logits = tf.concat([leak, logits], axis=-3)
    leaky_routing = cl.softmax(leaky_logits, axis=-3)
    return tf.split(leaky_routing, [1, leak_shape[-3]], axis=-3)[1]


def dynamicRouting_v1(votes,
                      num_routing=3,
                      use_bias=True,
                      leaky=True):
    """ Dynamic routing algorithm.

    See [Sabour et al., 2017](https://arxiv.org/abs/1710.09829).

    Args:
        votes: A 5-D or 7-D tensor with shape [batch_size, ..., in_channels/num_inputs, num_outputs] + out_caps_dims.
        num_routing: Integer, number of routing iterations.
        use_bias: Boolean, whether the layer uses a bias.
        leaky: Boolean, whether the algorithm uses leaky routing.

    Returns:
        poses: A 4-D or 6-D tensor.
        probs: A 2-D or 4-D tensor.
    """
    vote_shape = cl.shape(votes)
    logit_shape = vote_shape[:-2] + [1, 1]
    logits = tf.fill(logit_shape, 0.0)
    squash_on = -2 if vote_shape[-1] == 1 else [-2, -1]
    if use_bias:
        bias_shape = [1 for i in range(len(vote_shape) - 3)] + vote_shape[-3:]
        biases = tf.get_variable("biases",
                                 bias_shape,
                                 initializer=tf.constant_initializer(0.1),
                                 dtype=tf.float32)

    vote_stopped = tf.stop_gradient(votes, name="stop_gradient")
    for i in range(num_routing):
        with tf.variable_scope("iter_" + str(i)):
            if leaky:
                route = _leaky_routing(logits)
            else:
                route = cl.softmax(logits, axis=-3)
            if i == num_routing - 1:
                if use_bias:
                    preactivate = cl.reduce_sum(tf.multiply(route, votes), axis=-4, keepdims=True) + biases
                else:
                    preactivate = cl.reduce_sum(tf.multiply(route, votes), axis=-4, keepdims=True)
                poses = cl.ops.squash(preactivate, axis=squash_on)
            else:
                if use_bias:
                    preactivate = cl.reduce_sum(tf.multiply(route, vote_stopped), axis=1, keepdims=True) + biases
                else:
                    preactivate = cl.reduce_sum(tf.multiply(route, vote_stopped), axis=1, keepdims=True)
                poses = cl.ops.squash(preactivate, axis=squash_on)
                logits += cl.reduce_sum(vote_stopped * poses, axis=-4, keepdims=True)

    poses = tf.squeeze(poses, axis=-4)
    probs = tf.norm(poses, axis=(-2, -1))
    return(poses, probs)


def EMRouting(votes,
              activation,
              num_routing,
              use_bias=True):
    """
    Args:
        votes: [batch_size, ..., in_channels/num_inputs, num_outputs] + out_caps_dims.
        activation: [batch_size, ..., in_channels/num_inputs]

    Returns:
        pose: [batch_size, num_outputs] + out_caps_dims
        activation: [batch_size, num_outputs]
    """
    vote_shape = cl.shape(votes)
    num_outputs = vote_shape[-3]
    out_caps_dims = vote_shape[-2:]

    shape = vote_shape[:-2] + [np.prod(out_caps_dims)]
    votes = tf.reshape(votes, shape=shape)
    activation = activation[..., tf.newaxis, tf.newaxis]
    log_activation = tf.log(activation)

    log_R = tf.log(tf.fill(vote_shape[:-2] + [1], 1.) / num_outputs)

    lambda_min = 0.001
    lambda_max = 0.006
    for t_iter in range(num_routing):
        # increase inverse temperature linearly: lambda = k * t_iter + lambda_min
        #  let k = (lambda_max - lambda_min) / num_routing
        # TODO: search for the better lambda_min and lambda_max
        inverse_temperature = lambda_min + (lambda_max - lambda_min) * t_iter / max(1.0, num_routing)
        with tf.variable_scope('M-STEP') as scope:
            if t_iter > 0:
                scope.reuse_variables()
            pose, log_var, log_activation_prime = M_step(log_R, log_activation, votes, lambda_val=inverse_temperature)
            # It's no need to do the `E-STEP` in the last iteration
            if t_iter == num_routing - 1:
                break
        with tf.variable_scope('E-STEP'):
            log_R = E_step(pose, log_var, log_activation_prime, votes)
    pose = tf.reshape(pose, shape=vote_shape[:-4] + [num_outputs] + out_caps_dims)
    activation = tf.reshape(tf.exp(log_activation_prime), shape=vote_shape[:-4] + [num_outputs])
    return pose, activation


def M_step(log_R, log_activation, vote, lambda_val=0.01):
    R_shape = tf.shape(log_R)
    log_R = log_R + log_activation

    R_sum_i = cl.reduce_sum(tf.exp(log_R), axis=-3, keepdims=True)
    log_normalized_R = log_R - tf.reduce_logsumexp(log_R, axis=-3, keepdims=True)

    pose = cl.reduce_sum(vote * tf.exp(log_normalized_R), axis=-3, keepdims=True)
    log_var = tf.reduce_logsumexp(log_normalized_R + cl.log(tf.square(vote - pose)), axis=-3, keepdims=True)

    beta_v = tf.get_variable('beta_v',
                             shape=[1 for i in range(len(pose.shape) - 2)] + [pose.shape[-2], 1],
                             initializer=tf.truncated_normal_initializer(mean=15., stddev=3.))
    cost = R_sum_i * (beta_v + 0.5 * log_var)

    beta_a = tf.get_variable('beta_a',
                             shape=[1 for i in range(len(pose.shape) - 2)] + [pose.shape[-2], 1],
                             initializer=tf.truncated_normal_initializer(mean=100.0, stddev=10))
    cost_sum_h = cl.reduce_sum(cost, axis=-1, keepdims=True)
    logit = lambda_val * (beta_a - cost_sum_h)
    log_activation = tf.log_sigmoid(logit)

    return(pose, log_var, log_activation)


def E_step(pose, log_var, log_activation, vote):
    normalized_vote = cl.divide(tf.square(vote - pose), 2 * tf.exp(log_var))
    log_probs = normalized_vote + cl.log(2 * np.pi) + log_var
    log_probs = -0.5 * cl.reduce_sum(log_probs, axis=-1, keepdims=True)
    log_activation_logit = log_activation + log_probs
    log_activation_logit = log_probs
    log_R = log_activation_logit - tf.reduce_logsumexp(log_activation_logit, axis=-2, keepdims=True)
    return log_R
