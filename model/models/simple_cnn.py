from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, MaxPooling2D, \
    Flatten, Dense, Dropout
from tensorflow.keras.models import Model


def simple_cnn(input_shape=None, dropout=0.2, classes=3, is_model=True):
    signal_input = Input(shape=input_shape)

    # 1st Conv2D
    x = Conv2D(8, (1, 1), strides=(1, 1), activation='relu')(signal_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 2nd Conv2D
    x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

    # 3rd Conv2D
    x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

    # 4th Conv2D
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

    # 5th Conv2D
    x = Conv2D(16, (1, 1), strides=(1, 1), activation='relu')(x)
    # Full connection layer
    x = Flatten()(x)

    if is_model:
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        out = Dense(classes, activation='softmax')(x)
        model = Model(signal_input, out, name='cnn')

    else:
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        out = Dropout(dropout)(x)
        model = Model(signal_input, out, name='cnn_layer')

    return model
