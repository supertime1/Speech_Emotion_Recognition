from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.layers import Conv1D

@keras_export('keras.layers.DepthwiseConv1D')
class DepthwiseConv1D(Conv1D):
  def __init__(self,
               kernel_size,
               strides=1,
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DepthwiseConv1D, self).__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        # autocast=False,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input_shape):
    if len(input_shape) < 3:
        raise ValueError('Inputs to `DepthwiseConv1D` should have rank 3. '
                       'Received input shape:', str(input_shape))
    input_shape = tensor_shape.TensorShape(input_shape)

    if self.data_format == 'channels_last':
        channel_axis = -1
    elif self.data_format == 'channels_first':
        channel_axis = 1

    if input_shape.dims[channel_axis].value is None:
        raise ValueError('The channel dimens ion of the inputs to '
                       '`DepthwiseConv1D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    depthwise_kernel_shape = (self.kernel_size[0],
                              input_dim,
                              self.depth_multiplier)

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name='depthwise_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)

    if self.use_bias:
      self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    if self.padding == 'causal':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())
    if self.data_format == 'channels_last':
      spatial_start_dim = 1
    else:
      spatial_start_dim = 2

    # Explicitly broadcast inputs and kernels to 4D.
    strides = self.strides * 2
    inputs = array_ops.expand_dims(inputs, spatial_start_dim)
    depthwise_kernel = array_ops.expand_dims(self.depthwise_kernel, 0)
    dilation_rate = (1,) + self.dilation_rate

    outputs = backend.depthwise_conv2d(
        inputs,
        depthwise_kernel,
        strides=strides,
        padding=self.padding,
        dilation_rate=dilation_rate,
        data_format=self.data_format)

    if self.use_bias:
      outputs = backend.bias_add(
          outputs,
          self.bias,
          data_format=self.data_format)

    outputs = array_ops.squeeze(outputs, [spatial_start_dim])

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      length = input_shape[2]
      out_filters = input_shape[1] * self.depth_multiplier
    elif self.data_format == 'channels_last':
      length = input_shape[1]
      out_filters = input_shape[2] * self.depth_multiplier

    length = conv_utils.conv_output_length(length, self.kernel_size,
                                           self.padding,
                                           self.strides)
    if self.data_format == 'channels_first':
      return (input_shape[0], out_filters, length)
    elif self.data_format == 'channels_last':
      return (input_shape[0], length, out_filters)

  def get_config(self):
    config = super(DepthwiseConv1D, self).get_config()
    config.pop('filters')
    config.pop('kernel_initializer')
    config.pop('kernel_regularizer')
    config.pop('kernel_constraint')
    config['depth_multiplier'] = self.depth_multiplier
    config['depthwise_initializer'] = initializers.serialize(
        self.depthwise_initializer)
    config['depthwise_regularizer'] = regularizers.serialize(
        self.depthwise_regularizer)
    config['depthwise_constraint'] = constraints.serialize(
        self.depthwise_constraint)
    return config