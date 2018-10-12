# CapsLayer: An advanced library for capsule theory

Capsule theory is a potential research proposed by Geoffrey E. Hinton et al, where he describes the shortcomings of the Convolutional Neural Networks and how Capsules could potentially circumvent these problems such as "pixel attack" and create more robust Neural Network Architecture based on Capsules Layer.

We expect that this theory will definitely contribute to Deep Learning Industry and we are excited about it. For the same reason we are proud to introduce **CapsLayer**, an advanced library for the Capsule Theory, integrating capsule-relevant technologies, providing relevant analysis tools, developing related application examples, and probably most important thing: promoting the development of capsule theory. 

This library is based on [Tensorflow](https://www.tensorflow.org) and has a similar API with it but designed for capsule layers/models.


# Features

- TensorFlow-like API for building Neural Nets block, see [API docs](https://github.com/naturomics/CapsLayer/blob/master/docs/api_docs.md) for more details:
	- [x] capslayer.layers.conv2d
	- [x] capslayer.layers.conv1d
	- [x] capslayer.layers.fully_connected/dense
	- [x] capslayer.layers.primaryCaps
	- [x] capslayer.losses.spread_loss
	- [x] capslayer.losses.margin_loss

- Datasets support:
  - [x] [MNIST](http://yann.lecun.com/exdb/mnist)
  - [x] [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  - [x] [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
  - [ ] [small NORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small)

- Capsule Nets Model examples:
	- [x] [Dynamics routing between capsules](https://arxiv.org/abs/1710.09829)
	- [x] [matrix capsule with EM routing](https://openreview.net/forum?id=HJWLfGWRb)

- Algorithm support:
	- [x] Routing-by-agreement: including EM Routing and Dynamic Routing

If you want us to support more features, let us know by opening Issues or sending E-mail to naturomics.liao@gmail.com


# Documentation
- [Installation](docs/installation.md)
- [Tutorials](docs/tutorials.md) for running CapsNet on supported dataset (MNIST/CIFAR10 etc.) or your own dataset, or building your network with Capsule Layer
- [Theoretical Analysis](docs/articles.md)


# Contributions
Feel free to send your pull request or open issues


# Citation
If you find it is useful, please cite our project by the following BibTex entry:
```
@misc{HuadongLiao2017,
title = {CapsLayer: An advanced library for capsule theory},
author = {Huadong Liao, Jiawei He},
year = {2017}
publisher = {GitHub},
journal = {GitHub Project},
howpublished = {\url{http://naturomics.com/CapsLayer}},
}
```

> **Note:**
> We are considering to write a paper for this project, but before that, please cite the above Bibtex entry if you find it helps.


# License
Apache 2.0 license.
