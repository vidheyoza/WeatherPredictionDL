from keras.layers import Input, Dense, Conv2D, Flatten, LSTM, MaxPooling2D, TimeDistributed, ConvLSTM2D, Convolution2D, UpSampling2D, BatchNormalization, Conv3D, concatenate
from keras.models import Sequential, Model
from keras import backend as K


# Basic CNN for single day analysis
def model_1(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape', 'accuracy'])

    return model


# ConvLSTM implementation, with exploration in Conv3D
# ConvLSTM2D helps in time-series analysis from images
# Conv3D helps in better feature extraction from final ConvLSTM layer output
def model_2(input_shape):
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=64,
                       kernel_size=(3, 3),
                       input_shape=input_shape,
                       padding='same',
                       return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=128,
                       kernel_size=(3, 3),
                       padding='same',
                       return_sequences=True,
                       dropout=0.5))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=128,
                       kernel_size=(1, 1),
                       padding='same',
                       return_sequences=True,
                       dropout=0.5))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1,
                   kernel_size=(5, 1, 1),
                   activation='sigmoid',
                   data_format='channels_last'))

    # seq.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    seq.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae', 'mape', 'accuracy'])

    return seq


# Our primary model
# Modified implementation of U-Net (as dataset is small)
# Other ideas taken from model_2
def model_3(input_shape):
    # input_shape = (None, 16, 8, 6)
    input = Input(input_shape)

    c1 = ConvLSTM2D(16, (3, 3), activation='relu', recurrent_activation='relu', return_sequences=True, padding='same')(input)  # 16,8
    c1 = ConvLSTM2D(16, (3, 3), activation='relu', recurrent_activation='relu', return_sequences=True, padding='same')(c1)
    c1 = BatchNormalization()(c1)

    donwsample1 = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    c2 = ConvLSTM2D(32, (3, 3), activation='relu', recurrent_activation='relu', return_sequences=True, padding='same')(donwsample1)  # 8,4
    c2 = ConvLSTM2D(32, (3, 3), activation='relu', recurrent_activation='relu', return_sequences=True, padding='same')(c2)
    c2 = BatchNormalization()(c2)

    downsample2 = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)

    c3 = ConvLSTM2D(64, (1, 1), activation='relu', recurrent_activation='relu', return_sequences=True, padding='same')(downsample2)  # 4,2
    c3 = ConvLSTM2D(64, (1, 1), activation='relu', recurrent_activation='relu', return_sequences=True, padding='same')(c3)
    c3 = BatchNormalization()(c3)

    upsample2 = TimeDistributed(UpSampling2D((2, 2)))(c3)  # 8,4

    merge2 = concatenate([c2, upsample2], axis=4)  # 8,4

    conv1 = ConvLSTM2D(64, (3, 3), activation='relu', recurrent_activation='relu', return_sequences=True, padding='same')(merge2)
    conv1 = ConvLSTM2D(32, (3, 3), activation='relu', recurrent_activation='relu', return_sequences=True, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)

    upsample1 = TimeDistributed(UpSampling2D((2, 2)))(conv1)  # 16,8

    merge1 = concatenate([c1, upsample1], axis=4)  # 16,8

    conv2 = ConvLSTM2D(16, (3, 3), activation='relu', recurrent_activation='relu', return_sequences=False, padding='same')(merge1)
    conv2 = BatchNormalization()(conv2)
    out = (Conv2D(1, (1, 1), activation='relu', padding='same'))(conv2)

    model = Model(input, out)

    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape', 'accuracy'])

    return model


# Hyper-parameter tuning in model_2
def model_4(input_shape):
    # input_shape = (5, 8, 20, 6)
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                         input_shape=input_shape, padding='same', return_sequences=True,
                         activation='relu', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='relu', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='relu', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False,
                         activation='relu', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                     activation='relu',
                     padding='same', data_format='channels_last'))
    
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape', 'accuracy'])

    return model
