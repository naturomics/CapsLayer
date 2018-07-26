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

import tensorflow as tf


def int64_feature(value):
    """Casts value to a TensorFlow int64 feature list."""
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Casts value to a TensorFlow bytes feature list."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
