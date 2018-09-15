# Installation
CapsLayer is build on the top of TensorFlow with its python API, so you need python to be installed first. 

## Step 1: Install dependencies
- TesorFlow 

Follow the [Tensorflow installation instructions](https://www.tensorflow.org/install)([for China](https://tensorflow.google.cn/install/))

- NumPy and SciPy

system-level installation:

```
$ sudo pip install numpy
$ sudo pip install scipy
```

  user-level installation:

```
$ pip install --user numpy
$ pip install --user scipy
```

## Step 2: Install CapsLayer
**2.1** Clone or download this repository:
```
$ git clone https://github.com/naturomics/CapsLayer.git (In this way you should have git installed)
```

**2.2** Install CapsLayer
```
$ cd path/to/CapsLayer
# Option 1, temporarily add the path of CapsLayer to your PYTHONPATH environment variable so you don't
# need to install capslayer as system-level library
$ export PYTHONPATH=path/to/CapsLayer:${PYTHONPATH}

# Option 2, install capslayer to a user-local path
$ python setup.py install --user

# Option 3, install capslayer to a system-level path
$ sudo python setup.py install
```

**2.3** Make sure it's installed correctly, or you will fail to import capslayer with such 'Module Not Found' error #27
```
$ echo 'import capslayer as cl'>/tmp/test.py
$ python /tmp/test.py
```
if no error comes up, congratulations! And Play it fun!


## Step 3: *Try your first CapsLayer program*

Follow the instructions in [the Model Section](https://github.com/naturomics/CapsLayer/tree/master/models) to run your first CapsLayer model.


# TODO

- [ ] Test on Windows
