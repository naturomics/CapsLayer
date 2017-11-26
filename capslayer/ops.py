import numpy as np
import tensorflow as tf

from capslayer.utils import reduce_sum
from capslayer.utils import softmax

epsilon = 1e-9

def squash(vector):
    '''Squashing function
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1]
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    squared_norm = reduce_sum(tf.square(vector), axis=-2, keepdims=True)
    scalar_factor = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + epsilon)
    return(scalar_factor * vector)


def routing(vote,
            activation=None,
            num_outputs=32,
            out_caps_shape=[4, 4],
            method='EMRouting',
            num_iter=3,
            regularizer=None):
    ''' Routing-by-agreement algorithm.
    Args:
        alias H = out_caps_shape[0]*out_caps_shape[1].

        vote: [batch_size, num_inputs, num_outputs, H].
        activation: [batch_size, num_inputs, 1, 1].
        num_outputs: ...
        out_caps_shape: ...
        method: method for updating coupling coefficients between vote and pose['EMRouting', 'DynamicRouting'].
        num_iter: the number of routing iteration.
        regularizer: A (Tensor -> Tensor or None) function; the result of applying it on a newly created variable
                will be added to the collection tf.GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.

    Returns:
        pose: [batch_size, 1, 1, num_outputs] + out_caps_shape.
        activation: [batch_size, 1, 1, num_outputs].
    '''
    vote_stopped = tf.stop_gradient(vote, name="stop_gradient")
    batch_size = vote.shape[0].value
    if method == 'EMRouting':
        shape = vote.get_shape().as_list()[:3] + [1]
        # R: [batch_size, num_inputs, num_outputs, 1]
        R = tf.constant(np.ones(shape, dtype=np.float32) / num_outputs)
        for t_iter in range(num_iter):
            with tf.variable_scope('M-STEP') as scope:
                if t_iter > 0:
                    scope.reuse_variables()
                # It's no need to do the `E-STEP` in the last iteration
                if t_iter == num_iter - 1:
                    pose, stddev, activation_prime = M_step(R, activation, vote)
                    break
                else:
                    pose, stddev, activation_prime = M_step(R, activation, vote_stopped)
            with tf.variable_scope('E-STEP'):
                R = E_step(pose, stddev, activation_prime, vote_stopped)
        pose = tf.reshape(pose, shape=[batch_size, 1, 1, num_outputs] + out_caps_shape)
        activation = tf.reshape(activation_prime, shape=[batch_size, 1, 1, -1])
        return(pose, activation)
    elif method == 'DynamicRouting':
        B = tf.constant(np.zeros([batch_size, vote.shape[1].value, num_outputs, 1, 1], dtype=np.float32))
        for r_iter in range(num_iter):
            with tf.variable_scope('iter_' + str(r_iter)):
                coef = softmax(B, axis=2)
                if r_iter == num_iter - 1:
                    s = reduce_sum(tf.multiply(coef, vote), axis=1, keepdims=True)
                    pose = squash(s)
                else:
                    s = reduce_sum(tf.multiply(coef, vote_stopped), axis=1, keepdims=True)
                    pose = squash(s)
                    shape = [batch_size, vote.shape[1].value, num_outputs] + out_caps_shape
                    pose = tf.multiply(pose, tf.constant(1., shape=shape))
                    B += tf.matmul(vote_stopped, pose, transpose_a=True)
        return(pose, activation)

    else:
        raise Exception('Invalid routing method!', method)


def M_step(R, activation, vote, lambda_val=0.9, regularizer=None):
    '''
    Args:
        alias H = out_caps_shape[0]*out_caps_shape[1]

        vote: [batch_size, num_inputs, num_outputs, H]
        activation: [batch_size, num_inputs, 1, 1]
        R: [batch_size, num_inputs, num_outputs, 1]
        lambda_val: ...

    Returns:
        pose & stddev: [batch_size, 1, num_outputs, H]
        activation: [batch_size, 1, num_outputs, 1]
    '''
    batch_size = vote.shape[0].value
    # line 2
    R = tf.multiply(R, activation)
    R_sum_i = tf.reduce_sum(R, axis=1, keepdims=True) + epsilon

    # line 3
    # mean: [batch_size, 1, num_outputs, H]
    pose = tf.reduce_sum(R * vote, axis=1, keepdims=True) / R_sum_i

    # line 4
    stddev = tf.sqrt(tf.reduce_sum(R * tf.square(vote - pose), axis=1, keepdims=True) / R_sum_i + epsilon)

    # line 5, cost: [batch_size, 1, num_outputs, H]
    H = vote.shape[-1].value
    beta_v = tf.get_variable('beta_v', shape=[batch_size, 1, pose.shape[2].value, H], regularizer=regularizer)
    cost = (beta_v + tf.log(stddev)) * R_sum_i

    # line 6
    beta_a = tf.get_variable('beta_a', shape=[batch_size, 1, pose.shape[2], 1], regularizer=regularizer)
    activation = tf.nn.sigmoid(lambda_val * (beta_a - tf.reduce_sum(cost, axis=3, keepdims=True)))

    return(pose, stddev, activation)


def E_step(pose, stddev, activation, vote):
    '''
    Args:
        alias H = out_caps_shape[0]*out_caps_shape[1]

        pose & stddev: [batch_size, 1, num_outputs, H]
        activation: [batch_size, 1, num_outputs, 1]
        vote: [batch_size, num_inputs, num_outputs, H]

    Returns:
        pose & var: [batch_size, 1, num_outputs, H]
        activation: [batch_size, 1, num_outputs, 1]
    '''
    # line 2
    var = tf.square(stddev)
    x = tf.reduce_sum(tf.square(vote - pose) / (2 * var), axis=-1, keepdims=True)
    peak_height = 1 / (tf.reduce_prod(tf.sqrt(2 * np.pi * var + epsilon), axis=-1, keepdims=True) + epsilon)
    P = peak_height * tf.exp(-x)

    # line 3
    R = tf.nn.softmax(activation * P, axis=2)
    return(R)
