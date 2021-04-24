from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, MaxPooling2D, \
    Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def simple_cnn(input_shape=None, dropout=0.5, regularition=0.0001,
               classes=8, is_model=True):
    signal_input = Input(shape=input_shape)

    # 1st Conv2D
    x = Conv2D(8, (3, 3), strides=(1, 1), activation='relu',
               kernel_regularizer=regularizers.l2(regularition))(signal_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(dropout)(x)
    # 2nd Conv2D
    x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu',
               kernel_regularizer=regularizers.l2(regularition))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(dropout)(x)

    # 3rd Conv2D
    x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
               kernel_regularizer=regularizers.l2(regularition))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(dropout)(x)

    # 4th Conv2D
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
               kernel_regularizer=regularizers.l2(regularition))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(dropout)(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
               kernel_regularizer=regularizers.l2(regularition))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(dropout)(x)

    # 5th Conv2D
    x = Conv2D(16, (1, 1), strides=(1, 1), activation='relu',
               kernel_regularizer=regularizers.l2(regularition))(x)
    ## Full connection layer
    x = Flatten()(x)

    if is_model:
        x = Dense(16, activation='relu',
                  kernel_regularizer=regularizers.l2(regularition))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        out = Dense(classes, activation='softmax')(x)
        model = Model(signal_input, out, name='cnn')

    else:
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        out = Dropout(dropout)(x)
        model = Model(signal_input, out, name='cnn_layer')

    return model
