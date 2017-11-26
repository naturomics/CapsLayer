import tensorflow as tf


def spread_loss(labels, logits, margin, regularizer=None):
    '''
    Args:
        labels: [batch_size, num_label, 1].
        logits: [batch_size, num_label, 1].
        margin: Integer or 1-D Tensor.
        regularizer: use regularization.

    Returns:
        loss: Spread loss.
    '''
    # a_target: [batch_size, 1, 1]
    a_target = tf.matmul(labels, logits, transpose_a=True)
    dist = tf.maximum(0., margin - (a_target - logits))
    loss = tf.reduce_mean(tf.square(tf.matmul(1 - labels, dist, transpose_a=True)))
    if regularizer is not None:
        regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += tf.reduce_mean(regularizer)
    return(loss)


def margin_loss():
    pass


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
