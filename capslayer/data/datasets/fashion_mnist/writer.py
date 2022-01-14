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
import numpy as np
import tensorflow as tf

from capslayer.data.utils.TFRecordHelper import int64_feature, bytes_feature


FASHION_MNIST_FILES = {
    'train': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
    'eval': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
    'test': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
}


def load_fashion_mnist(path, split):
    split = split.lower()
    image_file, label_file = [os.path.join(path, file_name) for file_name in FASHION_MNIST_FILES[split]]

    with open(image_file) as fd:
        images = np.fromfile(file=fd, dtype=np.uint8)
        images = images[16:].reshape(-1, 784).astype(np.float32)
        if split == "train":
            images = images[:55000]
        elif split == "eval":
            images = images[55000:]
    with open(label_file) as fd:
        labels = np.fromfile(file=fd, dtype=np.uint8)
        labels = labels[8:].astype(np.int32)
        if split == "train":
            labels = labels[:55000]
        elif split == "eval":
            labels = labels[55000:]
    return(zip(images, labels))


def encode_and_write(dataset, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for image, label in dataset:
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(
                                       feature={'image': bytes_feature(image_raw),
                                                'label': int64_feature(label),
                                                'height': int64_feature(28),
                                                'width': int64_feature(28),
                                                'depth': int64_feature(1)}))
            writer.write(example.SerializeToString())


def tfrecord_runner(path, force=True):
    train_set = load_fashion_mnist(path, 'train')
    eval_set = load_fashion_mnist(path, 'eval')
    test_set = load_fashion_mnist(path, 'test')

    train_set_outpath = os.path.join(path, "train_fashion_mnist.tfrecord")
    eval_set_outpath = os.path.join(path, "eval_fashion_mnist.tfrecord")
    test_set_outpath = os.path.join(path, "test_fashion_mnist.tfrecord")

    if not os.path.exists(train_set_outpath) or force:
        encode_and_write(train_set, train_set_outpath)
    if not os.path.exists(eval_set_outpath) or force:
        encode_and_write(eval_set, eval_set_outpath)
    if not os.path.exists(test_set_outpath) or force:
        encode_and_write(test_set, test_set_outpath)


if __name__ == '__main__':
    path = "models/data/fashion_mnist"
    tfrecord_runner(path)
