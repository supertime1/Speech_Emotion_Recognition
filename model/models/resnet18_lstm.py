from model.models.resnet18_1d import resnet18_1d
from tensorflow.keras.layers import Input, TimeDistributed, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model


def resnet18_lstm(Tx, n_a, n_s, input_image_size, classes=3):
    # define resnet
    resnet = resnet18_1d(input_shape=(input_image_size, 1), classes=classes, as_model=False)

    X_input = Input(shape=(Tx, input_image_size, 1))

    X = TimeDistributed(resnet)(X_input)
    X = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    X = Bidirectional(LSTM(n_s))(X)
    X = Dense(classes, activation='softmax')(X)

    model = Model(inputs=[X_input], outputs=X)

    return model