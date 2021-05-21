from tensorflow.keras.layers import Conv1D, Input, BatchNormalization, Activation, \
    Add, ZeroPadding1D, MaxPooling1D, AveragePooling1D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


def identity_block_18_1d(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    X_shortcut = X

    # First component of main path
    X = Conv1D(filters=F1, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block_18_1d(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv1D(filters=F1, kernel_size=f, strides=s, padding='valid',
               name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same',
               name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv1D(filters=F1, kernel_size=f, strides=s, padding='valid',
                        name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=2, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def resnet18_1d(input_shape=(750, 1), classes=1, as_model=False):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding1D(3)(X_input)

    # Stage 1
    X = Conv1D(64, 7, strides=2, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, strides=2)(X)

    # Stage 2
    X = identity_block_18_1d(X, 3, [64, 64], stage=2, block='a')
    X = identity_block_18_1d(X, 3, [64, 64], stage=2, block='b')

    # Stage 3
    X = convolutional_block_18_1d(X, f=3, filters=[128, 128], stage=3, block='a', s=2)
    X = identity_block_18_1d(X, 3, [128, 128], stage=3, block='b')

    # Stage 4
    X = convolutional_block_18_1d(X, f=3, filters=[256, 256], stage=4, block='a', s=2)
    X = identity_block_18_1d(X, 3, [256, 256], stage=4, block='b')

    # Stage 5
    X = convolutional_block_18_1d(X, f=3, filters=[512, 512], stage=5, block='a', s=2)
    X = identity_block_18_1d(X, 3, [512, 512], stage=5, block='b')

    # AVGPOOL
    X = AveragePooling1D(2, name="avg_pool")(X)

    # output layer
    X = Flatten()(X)

    if as_model:
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create modelzoo
    model = Model(inputs=X_input, outputs=X, name='ResNet18_1d')

    return model
