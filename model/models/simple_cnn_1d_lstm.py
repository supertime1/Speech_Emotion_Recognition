import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, BatchNormalization, MaxPooling1D, \
    Dropout, Flatten, TimeDistributed, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model


def simple_cnn_1d_lstm(input_shape=(3, 1250, 1), classes=3):
    cnn = tf.keras.Sequential([
        # 1st Conv1D
        Conv1D(8, 1, strides=1, activation='relu', input_shape=(1250, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        # 2nd Conv1D
        Conv1D(16, 3, strides=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        # 3rd Conv1D
        Conv1D(32, 3, strides=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        # 4th Conv1D
        Conv1D(64, 3, strides=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        # 5th Conv1D
        Conv1D(16, 1, strides=1, activation='relu'),
        BatchNormalization(),
        # Full connection layer
        Flatten()
    ])

    X_input = Input(shape=input_shape)
    X = TimeDistributed(cnn)(X_input)
    X = Bidirectional(LSTM(32, return_sequences=True))(X)
    X = Bidirectional(LSTM(16))(X)
    X = Dense(classes, activation='softmax')(X)

    model = Model(inputs=[X_input], outputs=X)

    return model
