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
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.datasets.cifar import load_batch

from capslayer.data.utils.TFRecordHelper import int64_feature, bytes_feature


URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
md5sum = 'eb9058c3a382ffc7106e4002c42a8d85'


def load_cifar100(split, path=None):
    if path is None:
        cache_path = os.path.join(os.path.expanduser('~'), ".capslayer")
        path = get_file('cifar-100-python', cache_dir=cache_path, file_hash=md5sum, origin=URL, untar=True)

    split = split.lower()
    if split == 'test':
        fpath = os.path.join(path, 'test')
        images, labels = load_batch(fpath, label_key='fine_labels')
    else:
        fpath = os.path.join(path, 'train')
        images, labels = load_batch(fpath, label_key='fine_labels')

        idx = np.arange(len(images))
        np.random.seed(201808)
        np.random.shuffle(idx)

        labels = np.reshape(labels, (-1, ))
        images = images[idx[:45000]] if split == "train" else images[idx[45000:]]
        labels = labels[idx[:45000]] if split == "train" else labels[idx[45000:]]
    images = np.reshape(images.transpose(0, 2, 3, 1), (-1, 3072)).astype(np.float32)
    labels = np.reshape(labels, (-1, )).astype(np.int32)

    return(zip(images, labels))


def encode_and_write(dataset, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for image, label in dataset:
            print(image.shape)
            exit()
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(
                                       feature={'image': bytes_feature(image_raw),
                                                'label': int64_feature(label)}))
            writer.write(example.SerializeToString())


def tfrecord_runner(path=None, force=True):
    train_set = load_cifar100(path=path, split='train')
    eval_set = load_cifar100(path=path, split='eval')
    test_set = load_cifar100(path=path, split='test')

    if path is None:
        path = os.path.join(os.path.expanduser('~'), ".capslayer", "datasets", "cifar100")
    if not os.path.exists(path):
        os.makedirs(path)

    train_set_outpath = os.path.join(path, "train_cifar100.tfrecord")
    eval_set_outpath = os.path.join(path, "eval_cifar100.tfrecord")
    test_set_outpath = os.path.join(path, "test_cifar100.tfrecord")

    if not os.path.exists(train_set_outpath) or force:
        encode_and_write(train_set, train_set_outpath)
    if not os.path.exists(eval_set_outpath) or force:
        encode_and_write(eval_set, eval_set_outpath)
    if not os.path.exists(test_set_outpath) or force:
        encode_and_write(test_set, test_set_outpath)


if __name__ == "__main__":
    data = load_cifar100(split='train')
    print(data)
