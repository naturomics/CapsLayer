### `cl.layers.dense(inputs, activation, num_outputs, out_caps_dims, routing_method='EMRouting', coordinate_addition=False, reuse=None, name=None)`

A fully connected capsule layer.

#### Args:

* <b>`inputs`</b>: A 4-D tensor with shape [batch_size, num_inputs] + in_caps_dims or [batch_size, in_height, in_width, in_channels] + in_caps_dims
* <b>`activation`</b>: [batch_size, num_inputs] or [batch_size, in_height, in_width, in_channels]
* <b>`num_outputs`</b>: Integer, the number of output capsules in the layer.
* <b>`out_caps_dims`</b>: A list with two elements, pose shape of output capsules.
* <b>`routing_method`</b>: One of 'EMRouting' or 'DynamicRouting', the method for updating coupling coefficients between votes and pose
* <b>`coordinate_addition`</b>: Boolean, whether use Coordinate Addition technique proposed by [Hinton etc al.](https://openreview.net/forum?id=HJWLfGWRb), only works when `routing_method` is EM Routing.

#### Returns:
* <b>`pose`</b>: A 4-D tensor with shape [batch_size, num_outputs] + out_caps_dims
* <b>`activation`</b>: [batch_size, num_outputs]


### `cl.layers.primaryCaps(inputs, filters, kernel_size, strides, out_caps_dims, method=None, name=None)`:

Primary capsule layer.

#### Args:

* <b>`inputs`</b>: [batch_size, in_height, in_width, in_channels].
* <b>`filters`</b>: Integer, the dimensionality of the output space.
* <b>`kernel_size`</b>: kernel_size
* <b>`strides`</b>: strides
* <b>`out_caps_dims`</b>: A list of 2 integers.
* <b>`method`</b>: the method of calculating probability of entity existence(logistic, norm, None)

#### Returns:

* <b>`pose`</b>: A 6-D tensor, [batch_size, out_height, out_width, filters] + out_caps_dims
* <b>`activation`</b>: A 4-D tensor, [batch_size, out_height, out_width, filters]

### `cl.layers.conv2d(inputs, activation, filters, out_caps_dims, kernel_size, strides, padding="valid", routing_method="EMRouting", name=None, reuse=None)`:
   
A 2D convolutional capsule layer.

#### Args:

* <b>`inputs`</b>: A 6-D tensor with shape [batch_size, in_height, in_width, in_channels] + in_caps_dims.
* <b>`activation`</b>: A 4-D tensor with shape [batch_size, in_height, in_width, in_channels].
* <b>`filters`</b>: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
* <b>`out_caps_dims`</b>: A tuple/list of 2 integers, specifying the dimensions of output capsule, e.g. out_caps_dims=[4, 4] representing that each output capsule has shape [4, 4].
* <b>`kernel_size`</b>:  An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimens
* <b>`strides`</b>: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dim
* <b>`padding`</b>: One of "valid" or "same" (case-insensitive), now only support "valid".
* <b>`routing_method`</b>: One of "EMRouting" or "DynamicRouting", the method of routing-by-agreement algorithm.
* <b>`name`</b>: A string, the name of the layer.
* <b>`reuse`</b>: Boolean, whether to reuse the weights of a previous layer by the same name.

#### Returns:

* <b>`pose`</b>: A 6-D tensor with shape [batch_size, out_height, out_width, out_channels] + out_caps_dims.
* <b>`activation`</b>: A 4-D tensor with shape [batch_size, out_height, out_width, out_channels].

### `cl.layers.conv3d(inputs, activation, filters, out_caps_dims, kernel_size, strides, padding="valid", routing_method="EMRouting", name=None, reuse=None)`:

A 3D convolutional capsule layer.

#### Args:

* <b>`inputs`</b>: A 7-D tensor with shape [batch_size, in_depth, in_height, in_width, in_channels] + in_caps_dims.
* <b>`activation`</b>: A 5-D tensor with shape [batch_size, in_depth, in_height, in_width, in_channels].
* <b>`filters`</b>: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
* <b>`out_caps_dims`</b>: A tuple/list of 2 integers, specifying the dimensions of output capsule, e.g. out_caps_dims=[4, 4] representing that each output capsule has shape [4, 4].
* <b>`kernel_size`</b>:  An integer or tuple/list of 3 integers, specifying the height and width of the 3D convolution window. Can be a single integer to specify the same value for all spatial dimens
* <b>`strides`</b>: An integer or tuple/list of 3 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dim
* <b>`padding`</b>: One of "valid" or "same" (case-insensitive), now only support "valid".
* <b>`routing_method`</b>: One of "EMRouting" or "DynamicRouting", the method of routing-by-agreement algorithm.
* <b>`name`</b>: String, a name for the operation (optional).
* <b>`reuse`</b>: Boolean, whether to reuse the weights of a previous layer by the same name.

#### Returns:

* <b>`pose`</b>: A 7-D tensor with shape [batch_size, out_depth, out_height, out_width, out_channels] + out_caps_dims.
* <b>`activation`</b>: A 5-D tensor with shape [batch_size, out_depth, out_height, out_width, out_channels].


### `cl.layers.conv1d(inputs, activation, filters, out_caps_dims, kernel_size, stride, padding="valid", routing_method="EMRouting", name=None, reuse=None)`:

A 1D convolutional capsule layer (e.g. temporal convolution).

#### Args:

* <b>`inputs`</b>: A 5-D tensor with shape [batch_size, in_width, in_channels] + in_caps_dims.
* <b>`activation`</b>: A 3-D tensor with shape [batch_size, in_width, in_channels].
* <b>`kernel_size`</b>: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
* <b>`strides`</b>: An integer or tuple/list of a single integer, specifying the stride length of the convolution.

#### Returns:

* <b>`pose`</b>: A 5-D tensor with shape [batch_size, out_width, out_channels] + out_caps_dims.
* <b>`activation`</b>: A 3-D tensor with shape [batch_size, out_width, out_channels].


### `cl.losses.spread_loss(labels, logits, margin, regularizer=None)`:

#### Args:

* <b>`labels`</b>: [batch_size, num_label].
* <b>`logits`</b>: [batch_size, num_label].
* <b>`margin`</b>: Integer or 1-D Tensor.
* <b>`regularizer`</b>: use regularization.

#### Returns:

* <b>`loss`</b>: Spread loss.


### `cl.losses.margin_loss(labels, logits, upper_margin=0.9, bottom_margin=0.1, downweight=0.5)`:

#### Args:

* <b>`labels`</b>: [batch_size, num_label].
* <b>`logits`</b>: [batch_size, num_label]
