As a basic network block, capsule layer can be an alternative of CNN block, therefore we design CapsLayer as a high level library so that the community can use its APIs to build their own neural network without making wheels again. With this consider, [capslayer](https://github.com/naturomics/CapsLayer/tree/master/capslayer) is a package focusing on the core algorithm and auxiliary tools of capsule and providing some TF-like APIs for calling, while [models](https://github.com/naturomics/CapsLayer/tree/master/models) is a model zoo implementing the network architectures proposed in capsule-relevant papers with capslayer APIs.

In this tutorial, we are going to tell you how to:

1. Play CapsNet with supported dataset;
2. Apply your own dataset to CapsNet;
3. Build your own capsule neural networks with CapsLayer

## 1. Play the CapsNets zoo
Please follow the instructions [here](https://github.com/naturomics/CapsLayer/tree/master/models)

## 2. Apply your dataset to CapsNets
For training/testing a deep learning model, it always follows these steps: 1. reading data; 2. preprocessing data if necessary; and 3. training/test model. A well-designed project should modularize these processes so that we don't need to change too much code of model to apply to a new dataset, on the contrary we also don't need to change too much in data reading/preprocessing pipeline to apply the same dataset to a new model. In the model zoo of CapsLayer, we achieve it by designing a DataLoader object so that training/testing pipeline only communicate with this object to get data in, see the [MNIST demo](https://github.com/naturomics/CapsLayer/blob/master/capslayer/data/datasets/mnist/reader.py#L47).

Therefore, if you are trying to apply your dataset to CapsNet, you can follow our MNIST demo:

1. a [writer](https://github.com/naturomics/CapsLayer/blob/master/capslayer/data/datasets/mnist/writer.py) for preprocessing dataset and saving the preprocessed result to .tfrecord or .npy files. If you are new to Tensorflow or programming, please read the tutorials in Tensorflow offical website or search online for how to make .tfrecord/.npy file. The [MNIST writer](https://github.com/naturomics/CapsLayer/blob/master/capslayer/data/datasets/mnist/writer.py) can be an example for you.

2. a [reader](https://github.com/naturomics/CapsLayer/blob/master/capslayer/data/datasets/mnist/reader.py) defines a DataLoader object for communicating with the training/testing pipeline. In the reader, we use TF's high-level API **[tf.data](https://tensorflow.google.cn/guide/datasets)** for importing data, follow the link to learn how it works. The basic idea is using different Iterator for different runing phase: one-shot iterator with using data iteratively(epoch) for training phase, and initializable iterator for validation/test phase. Note that a **[handle](https://github.com/naturomics/CapsLayer/blob/63de13f7be8a6986485988b5405efdc55539cdac/capslayer/data/datasets/mnist/reader.py#L75)** and **[next_element](https://github.com/naturomics/CapsLayer/blob/63de13f7be8a6986485988b5405efdc55539cdac/capslayer/data/datasets/mnist/reader.py#L108)** member in DataLoader are import which will be called in main.py script.

3. Decide which dataset to use in the [main.py](https://github.com/naturomics/CapsLayer/blob/master/models/main.py) script so the corresponding DataLoader object will be imported. You need to do some changes to suit your dataset:
- modify the parameters such as image height, width, and number of labels [here](https://github.com/naturomics/CapsLayer/blob/63de13f7be8a6986485988b5405efdc55539cdac/models/main.py#L205)
- import your DataLoader object [here](https://github.com/naturomics/CapsLayer/blob/63de13f7be8a6986485988b5405efdc55539cdac/models/main.py#L220)

That's all you need to do for applying your own dataset to CapsNet. If your custom DataLoader is right, it should work now, you can refer [these steps](https://github.com/naturomics/CapsLayer/tree/master/models) to run your model.

## 3. Build your capsule NN with [CapsLayer](https://github.com/naturomics/CapsLayer)
As we mentioned previously, capsule layer can be an alternative of CNN block, meaning you can use CapsLayer to design a new capsule neural network. For this purpose, you can use CapsLayer APIs in a similar way with TensorFlow APIs, see our [API doc](https://github.com/naturomics/CapsLayer/blob/master/docs/api_docs.md) for what we provide.
