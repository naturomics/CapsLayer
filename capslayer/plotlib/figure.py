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
from matplotlib import pyplot as plt
from matplotlib import ticker


def plot_activation(matrix, step, save_to=None):
    save_to = os.path.join(".", "activations") if save_to is None else save_to
    os.makedirs(save_to, exist_ok=True)
    if len(matrix.shape) != 2:
        raise ValueError('Input "matrix" should have 2 rank, but it is',str(len(matrix.shape)))
    num_label = matrix.shape[1] - 1
    matrix = matrix[matrix[:, num_label].argsort()]
    fig, axes = plt.subplots(ncols=1, nrows=num_label, figsize=(15,12))
    fig.suptitle("The probability of entity presence (step %s)"%str(step), fontsize=20)
    fig.tight_layout()
    for i, ax in enumerate(axes.flatten()):
        idx = num_label - (i + 1)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Capsule " + str(idx))
        ax.yaxis.set_major_locator(ticker.NullLocator())
        if idx > 0:
            ax.xaxis.set_major_locator(ticker.NullLocator())
        else:
            ax.xaxis.set_major_locator(ticker.IndexLocator(base=500,offset=0))
            ax.set_xlabel("Sample index ")
        ax.plot(matrix[:,idx])
        ax_prime = ax.twinx()
        ax_prime.spines['top'].set_color('none')
        ax_prime.spines['bottom'].set_color('none')
    plt.subplots_adjust(hspace=0.2, left=0.05, right=0.95, bottom=0.05, top=.95)
    plt.savefig(os.path.join(save_to, "activation_%s.png" % str(step)))
    plt.close()
