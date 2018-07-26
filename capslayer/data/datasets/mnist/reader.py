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

import os
import tensorflow as tf
from capslayer.data.utils.download_utils import maybe_download_and_extract
from capslayer.data.datasets.mnist.writer import tfrecord_runner


def parse_fun(serialized_example):
    """ Data parsing function.
    """
    features = tf.parse_single_example(serialized_example,
                                       features={'image': tf.FixedLenFeature([], tf.string),
                                                 'label': tf.FixedLenFeature([], tf.int64),
                                                 'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'depth': tf.FixedLenFeature([], tf.int64)})
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, shape=[height * width * depth])
    image.set_shape([28 * 28 * 1])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)
    features = {'images': image, 'labels': label}
    return(features)


class DataLoader(object):
    """ Data Loader.
    """
    def __init__(self, path=None,
                 num_works=1,
                 splitting="TVT",
                 one_hot=False,
                 name="create_inputs"):
        """
        Args:
            path: Path to store data.
            name: Name for the operations.
        """

        # path exists and is writable?
        if path is None:
            path = os.path.join(os.environ["HOME"], ".cache", "capslayer", "datasets", "mnist")
            os.makedirs(path, exist_ok=True)
        elif os.access(path, os.F_OK):
            path = path if os.path.basename(path) == "mnist" else os.path.join(path, "mnist")
            os.makedirs(path, exist_ok=True)
        elif os.access(path, os.W_OK):
            raise IOError("Permission denied! Path %s is not writable." % (str(path)))

        # data downloaded and data extracted?
        maybe_download_and_extract("mnist", path)
        # data tfrecorded?
        tfrecord_runner(path, force=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = None
        self.path = path
        self.name = name

    def __call__(self, batch_size, mode):
        """
        Args:
            batch_size: Integer.
            mode: Running phase, one of "train", "test" or "eval"(only if splitting='TVT').
        """
        with tf.name_scope(self.name):
            mode = mode.lower()
            modes = ["train", "test", "eval"]
            if mode not in modes:
                raise "mode not found! supported modes are " + modes
            filenames = [os.path.join(self.path, "%s_mnist.tfrecord" % mode)]
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(parse_fun)
            dataset = dataset.batch(batch_size)

            if mode == "train":
                dataset = dataset.shuffle(buffer_size=50000)
                dataset = dataset.repeat()
                iterator = dataset.make_one_shot_iterator()
            elif mode == "eval":
                dataset = dataset.repeat(1)
                iterator = dataset.make_initializable_iterator()
            elif mode == "test":
                dataset = dataset.repeat(1)
                iterator = dataset.make_one_shot_iterator()

            if self.next_element is None:
                self.next_element = tf.data.Iterator.from_string_handle(self.handle, iterator.output_types).get_next()

            return(iterator)
