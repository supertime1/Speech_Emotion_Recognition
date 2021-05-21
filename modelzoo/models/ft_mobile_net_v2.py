from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


def FT_Mobile_Net_V2(input_shape, num_classes, norm_layer, dropout, reshape=128):
    inputs = Input(shape=input_shape)
    #x = preprocessing.Resizing(reshape, reshape)(inputs)
    x = norm_layer(inputs)
    # rescale images pixel values from [0-255] to [-1,1], which is expected by MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')
    # whether to train top layers (first layers)
    base_model.trainable = False
    # whether to train batch normalization layer
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes)(x)
    model = Model(inputs, outputs, name='ft_mobile_net_v2')
    return model
