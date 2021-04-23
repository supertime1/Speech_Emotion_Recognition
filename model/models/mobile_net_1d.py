from model.layers.depthwiseconv1d import DepthwiseConv1D
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, \
    GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.models import Model


def MobileNet_1D(input_shape=None, dropout=0.2, alpha=1, classes=3, is_model=True):
    signal_input = Input(shape=input_shape)

    x = Conv1D(int(32 * alpha), 3, strides=2, padding='same', use_bias=False)(signal_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(64 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(128 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(128 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(256 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(256 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(512 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(1024 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(1024 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)

    if is_model:
        out = Dense(classes, activation='softmax')(x)
        model = Model(signal_input, out, name='mobilenet_1d')
    else:
        out = Dense(64, activation='relu')(x)
        model = Model(signal_input, out, name='mobilenet_1d')

    return model
