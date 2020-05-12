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
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

from nn_ops import space_to_batch_nd_v1
from nn_ops import space_to_batch_nd
from nn_ops import batch_to_space_nd


def testSpaceToBatch():
    shape = [1, 9, 9, 9]
    input_tensor = tf.reshape(tf.range(np.prod(shape), dtype=tf.float32), shape=shape)
    return space_to_batch_nd(input_tensor, kernel_size=(3, 3, 3), strides=(1, 1, 1))


def testSpaceToBatch_v1():
    shape = [1, 9, 9, 9]
    input_tensor = tf.reshape(tf.range(np.prod(shape), dtype=tf.float32), shape=shape)
    return space_to_batch_nd_v1(input_tensor, kernel_size=(3, 3, 3), strides=(1, 1, 1))


def testBatchToSpace(input):
    return batch_to_space_nd(input, spatial_shape=[7, 7, 7])

if __name__ == "__main__":
    transfered = testSpaceToBatch()
    transfered_v1 = testSpaceToBatch_v1()
    out = testBatchToSpace(tf.reduce_mean(transfered, axis=-1))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()
    out1 = sess.run(transfered)
    out2 = sess.run(transfered_v1)
    # out2 = sess.run(out)
    print(out1[2, :])
    print(out1[2, :])
    # print(out1[1, 18:])
    # print(out2[0])
    # print(np.mean(out1[1]))
    # print(out2)
    # print(out)
