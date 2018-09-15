# CapsLayer Models

This module contains a number of different models which are already public:

- The [dynamic routing CapsNet](https://arxiv.org/abs/1710.09829)

- The [EM routing matrix CapsNet](https://openreview.net/forum?id=HJWLfGWRb)

To use these models, please follow the below instructions


**Step 1.** Install CapsLayer follow [this instructions](https://github.com/naturomics/CapsLayer/blob/master/docs/installation.md)

**Step 2.** Training your model
```
$ cd CapsLayer
$ python models/main.py --dataset <name_of_dataset> [--batch_size=32, ...]
```

run `python models/main.py --helpfull` to print helps.

**Step 3.** Visualize the results with `tensorboard` or tools provided by this project. The following is an example plotted with our `cl.plotlib` tool:
![activation_map](assets/results_mnist_vecCapsNetactivations.gif)

**Step 5.** Test your model using the same command line as `step 2` but with paremeter `--nois_training`, like this:
```
$ python models/main.py --dataset <name_of_dataset> --nois_training False [--batch_size=32, ...]
```

If you want to apply these models to the datasets that have not yet been supported, please see [the tutorial](https://github.com/naturomics/CapsLayer/blob/master/docs/tutorials.md) for instructions on how to modify them.
