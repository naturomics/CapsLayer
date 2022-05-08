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


def testConv2d():
    shape = [1, 9, 9, 9, 1, 1]
    input_tensor = tf.random_normal(shape=shape, stddev=5, seed=2018)
    activation = tf.random_normal(shape=shape[:4], mean=0.5, stddev=0.5, seed=2018)
    activation = tf.clip_by_value(activation, 0., 0.999)
    conv_cl, activation = cl.layers.conv2d(input_tensor,
                                           activation,
                                           filters=1,
                                           kernel_size=3,
                                           strides=1,
                                           out_caps_dims=[1, 1])
    input_tensor = tf.squeeze(input_tensor, axis=(-2, -1))
    conv_tf = tf.layers.conv2d(input_tensor, filters=1, kernel_size=3, strides=1, use_bias=False)

    return tf.squeeze(conv_cl, axis=(-2, -1)), conv_tf


if __name__ == "__main__":
    conv_cl, conv_tf = testConv2d()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    conv_cl = sess.run(conv_cl)
    conv_tf = sess.run(conv_tf)
    print(conv_cl)
    # print(conv_tf)
