from tensorflow.keras.layers import Conv1D, Input, BatchNormalization, MaxPooling1D, \
    Dropout, Flatten, Dense
from tensorflow.keras.models import Model


def simple_cnn_1d(input_shape=None, dropout=0.2, classes=3, is_model=True):
    signal_input = Input(shape=input_shape)

    x = Conv1D(8, 1, strides=1, activation='relu')(signal_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Dropout(dropout)(x)
    # 2nd Conv1D
    x = Conv1D(16, 3, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Dropout(dropout)(x)
    # 3rd Conv1D
    x = Conv1D(32, 3, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Dropout(dropout)(x)
    # 4th Conv1D
    x = Conv1D(64, 3, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Dropout(dropout)(x)
    # 5th Conv1D
    x = Conv1D(16, 1, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)

    # Full connection layer
    x = Flatten()(x)

    if is_model:
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        out = Dense(classes, activation='softmax')(x)
        model = Model(signal_input, out, name='cnn_1d')

    else:
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        out = Dropout(dropout)(x)
        model = Model(signal_input, out, name='cnn_1d_layer')

    return model
