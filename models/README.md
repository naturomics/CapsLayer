# CapsLayer Models

This module contains a number of different models which are already public:

The [dynamic routing CapsNet](https://arxiv.org/abs/1710.09829)

The [EM routing matrix CapsNet](https://arxiv.org/abs/1710.09829)

To use these models, please follow the following instructions


**Step 1.** Clone this repository with `git`:
```
$ git clone https://github.com/naturomics/CapsLayer.git
$ cd CapsLayer
```

**Step 2.** Download dataset with the script `download_data.py`(not completed yet)
```
$ python models/tools/download_data.py --dataset mnist/fashion-mnist
```

**Step 3.** Training your model
```
$ python models/main.py --dataset mnist/fashion-mnist [--batch_size=128]
```

**Step 4.** Visualize the results with `tensorboard` or tools provided by this project. The following is an example plotted with our `plot_activation.R` tool:
![activation_map](assets/results_mnist_vecCapsNetactivations.gif)

If you want to apply these models to the datasets that have not yet been supported, please see [the tutorial](https://github.com/naturomics/CapsLayer/blob/master/docs/tutorials.md) for instructions on how to modify them
