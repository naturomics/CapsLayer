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
import sys
import gzip
import shutil
from six.moves import urllib

# mnist dataset
HOMEPAGE = "http://yann.lecun.com/exdb/mnist/"
MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

# fashion-mnist dataset
HOMEPAGE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
FASHION_MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
FASHION_MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
FASHION_MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

# smallNORB dataset
HOMEPAGE = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
SMALLNORB_TRAIN_DAT_URL = HOMEPAGE + "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz"
SMALLNORB_TRAIN_CAT_URL = HOMEPAGE + "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz"
SMALLNORB_TRAIN_INFO_URL = HOMEPAGE + "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz"
SMALLNORB_TEST_DAT_URL = HOMEPAGE + "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz"
SMALLNORB_TEST_CAT_URL = HOMEPAGE + "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz"
SMALLNORB_TEST_INFO_URL = HOMEPAGE + "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz"

# CIFAR-10 dataset
HOMEPAGE = "http://www.cs.toronto.edu/~kriz/"
CIFAR_10_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_10_BIN_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

# CIFAR-100 dataset
HOMEPAGE = "http://www.cs.toronto.edu/~kriz/"
CIFAR_100_URL = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR_100_BIN_URL = "http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"


def download_and_uncompress_zip(URL, dataset_dir, force=False):
    '''
    Args:
        URL: the download links for data
        dataset_dir: the path to save data
        force: redownload data
    '''
    filename = URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    extract_to = os.path.splitext(filepath)[0]

    def download_progress(count, block_size, total_size):
        sys.stdout.write("\r>> Downloading %s %.1f%%" % (filename, float(count * block_size) / float(total_size) * 100.))
        sys.stdout.flush()

    if not force and os.path.exists(filepath):
        print("file %s already exist" % (filename))
        return 0
    else:
        filepath, _ = urllib.request.urlretrieve(URL, filepath, download_progress)
        print()
        print('Successfully Downloaded', filename)

    # with zipfile.ZipFile(filepath) as fd:
    with gzip.open(filepath, 'rb') as f_in, open(extract_to, 'wb') as f_out:
        print('Extracting ', filename)
        shutil.copyfileobj(f_in, f_out)
        print('Successfully extracted')
        print()


def maybe_download_and_extract(dataset, save_to, force=False):
    if dataset == 'mnist':
        print("Start downloading dataset MNIST:")
        download_and_uncompress_zip(MNIST_TRAIN_IMGS_URL, save_to, force)
        download_and_uncompress_zip(MNIST_TRAIN_LABELS_URL, save_to, force)
        download_and_uncompress_zip(MNIST_TEST_IMGS_URL, save_to, force)
        download_and_uncompress_zip(MNIST_TEST_LABELS_URL, save_to, force)
    elif dataset == 'fashion-mnist' or dataset == 'fashion_mnist':
        print("Start downloading dataset Fashion MNIST:")
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_IMGS_URL, save_to, force)
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_LABELS_URL, save_to, force)
        download_and_uncompress_zip(FASHION_MNIST_TEST_IMGS_URL, save_to, force)
        download_and_uncompress_zip(FASHION_MNIST_TEST_LABELS_URL, save_to, force)
    elif dataset == 'smallNORB':
        print("Start downloading dataset small NORB:")
        download_and_uncompress_zip(SMALLNORB_TRAIN_DAT_URL, save_to, force)
        download_and_uncompress_zip(SMALLNORB_TRAIN_CAT_URL, save_to, force)
        download_and_uncompress_zip(SMALLNORB_TRAIN_INFO_URL, save_to, force)
        download_and_uncompress_zip(SMALLNORB_TEST_DAT_URL, save_to, force)
        download_and_uncompress_zip(SMALLNORB_TEST_CAT_URL, save_to, force)
        download_and_uncompress_zip(SMALLNORB_TEST_INFO_URL, save_to, force)
    elif dataset == 'cifar-10' or dataset == 'cifar10' or dataset == "cifar_10":
        download_and_uncompress_zip(CIFAR_10_BIN_URL, save_to, force)
    elif dataset == 'cifar-100' or dataset == 'cifar100' or dataset == "cifar_100":
        download_and_uncompress_zip(CIFAR_100_URL, save_to, force)
    else:
        raise Exception("Invalid dataset name! please check it: ", dataset)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for automatically downloading datasets')
    parser.add_argument("--dataset", default='mnist', choices=['mnist', 'fashion-mnist', 'smallNORB', 'cifar-10', 'cifar-100'])
    parser.add_argument("--save_to", default='models/data/mnist')
    parser.add_argument("--force", default=False, type=bool)
    args = parser.parse_args()
    maybe_download_and_extract(args.dataset, args.save_to, args.force)
