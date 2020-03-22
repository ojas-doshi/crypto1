from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Reshape, InputLayer
from tensorflow.keras.optimizers import RMSprop
import os

def get_model():
    model = Sequential()

    model.add(InputLayer(input_shape=(30,)))
    model.add(Reshape((1, 30)))

    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))

    model.add(Dense(25, activation='relu'))

    model.add(Dense(1, activation='relu'))

    optimizer1 = RMSprop(lr=.0001)  
    model.compile(loss='mse', optimizer=optimizer1, metrics=['accuracy', 'mae'])

    return model


def directory1(date1,model):
    os.mkdir(date1)
    save_model(model,''+date1+'/model.h5')
    