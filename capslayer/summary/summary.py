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


def image(name,
          tensor,
          verbose=True,
          max_outputs=3,
          collections=None,
          family=None):
    if verbose:
        tf.summary.image(name, tensor, max_outputs, collections, family)
    else:
        pass


def scalar(name, tensor, verbose=False, collections=None, family=None):
    if verbose:
        tf.summary.scalar(name, tensor, collections, family)
    else:
        pass


def histogram(name, values, verbose=False, collections=None, family=None):
    if verbose:
        tf.summary.histogram(name, values, collections, family)
    else:
        pass


def tensor_stats(name, tensor, verbose=True, collections=None, family=None):
    """
    Args:
        tensor: A non-scalar tensor.
    """
    if verbose:
        with tf.name_scope(name):
            mean = tf.reduce_mean(tensor)
            tf.summary.scalar('mean', mean, collections=collections, family=family)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
            tf.summary.scalar('stddev', stddev, collections=collections, family=family)
            tf.summary.scalar('max', tf.reduce_max(tensor), collections=collections, family=family)
            tf.summary.scalar('min', tf.reduce_min(tensor), collections=collections, family=family)
            tf.summary.histogram('histogram', tensor, collections=collections, family=family)
    else:
        pass
