# CapsLayer: An advanced library for capsule theory

Capsule theory is a potential research proposed by Geoffrey E. Hinton et al, where he describes the shortcomings of the Convolutional Neural Networks and how Capsules could potentially curcumvent these problems such as "pixel attack" and create more robust Neural Network Architecture baded on Capsules Layer.

We expect that this theory will definitely contribute to Deep Learning Industry and we are excited about it. For the same reason we are proud to introduce **CapsLayer**, an advanced library for the Capsule Theory, integrating capsule-relevant technologies, providing relevant analysis tools, developing related application examples, and probably most important thing: promoting the development of capsule theory. 

This library is based on [Tensorflow](www.tensorflow.org) and has a similar API with it but designed for capsule layer/model. We will soon be testing it with TensorFlow 1.4.x as well as TensorFlow 1.5.x which introduces several imperative modules such as Eager Execution etc.


# Features

- TensorFlow-like API for building Neural Nets block:
	- [x] capslayer.layers.conv2d
	- [x] capslayer.layers.fully_connected
	- [x] capslayer.layers.primaryCaps
	- [x] capslayer.losses.spread_loss
	- [ ] capslayer.losses.margin_loss

- Datasets support:
  - [x] [MNIST](http://yann.lecun.com/exdb/mnist)
  - [x] [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  - [ ] [small NORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small)
  - [ ] [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)

- Capsule Nets Model examples:
	- [x] [Dynamics routing between capsules](https://arxiv.org/abs/1710.09829)
	- [ ] [matrix capsule with EM routing](https://openreview.net/pdf?id=HJWLfGWRb)(will be released soon)

- Algorithm support:
	- [x] Routing-by-agreement: including EM Routing and Dynamic Routing
	- [ ] [Non-local Neural Networks](https://arxiv.org/abs/1711.07971)

If you want us to support more features, please tell us by opening an Issue or sending E-mail to `naturomics.liao@gmail.com`


# Documentation
- [Installation](docs/installation.md)
- [Tutorials](docs/tutorials.md)
- [Theoretical Analysis](docs/article.md)


# Contributions
Feel free to send your pull request or open an issue


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

> **Note**
> We are considering to write a paper for this project, but before that, cite the above Bibtex entry.


# License
Apache 2.0 license.
