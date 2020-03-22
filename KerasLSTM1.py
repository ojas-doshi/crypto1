from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Reshape, InputLayer
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard
from numpy import genfromtxt
from datetime import datetime
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os 
import io
from tensorflow import lite



date = str(datetime.now()).replace(':','_')


# model.save(''+date+'/model'+date+'.h5')

# predt = predictions*(maxval-minval)+minval
# print(Y[0][0]*(maxval-minval)+minval,predt[0][0])
# ax.scatter(X,Y, predt)
# model2=load_model(''+date+'/model'+date+'.h5')
# model2=super1.get_model()
# model2.load_weights(''+date+'/model'+date+'.h5')
# model2.summary()





class KerasLSTM():
    def get_model(self):
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


    def plot_graph(self):
        Y.shape
        # predt.shape
        x = X[:,:1]
        #plt.style.use()
        fig,ax = plt.subplots(nrows=2,ncols=2)
        print(ax)
        ax.plot(x,Y,'k--')
        # ax.plot(x,predt)
        ax.set_xlabel('Original')
        ax.set_ylabel('Predicted')
        plt.savefig(''+date+'/plot'+date+'.png')
        # plt.show()
        # plt.legend()


    def predictions(self):
        predictions = model.predict(X)

        print(predictions[0][0]*(maxval-minval)+minval)
        print(predictions[1][0]*(maxval-minval)+minval)
        print(predictions[2][0]*(maxval-minval)+minval)
        print(predictions[3][0]*(maxval-minval)+minval)

    def save__model(self):
        # os.mkdir(date)
        save_model(model,'model1.h5')

        # conv = lite.TFLiteConverter.from_saved_model('savedmodel')
        conv = lite.TFLiteConverter.from_keras_model_file('model1.h5')
        tfmodel = conv.convert()
        open ("model.tflite" , "wb") .write(tfmodel)

        # conv1 = lite.TFLiteConverter.from_keras_model_file('model.h5')
        # tfmodel1 = conv1.convert()
        # open("model1.tflite","wb").write(tfmodel1)


kl = KerasLSTM()
dataset_original = genfromtxt('LTC2.csv', delimiter=',')

dataset_original = dataset_original[:]

minval=dataset_original.min()
maxval=dataset_original.max()

dataset= (dataset_original-minval)/(maxval-minval)

X = dataset[:, 0:30]
Y = dataset[:, 30:]

print("X", X.shape)
print("Y", Y.shape)

start = timer()

model = kl.get_model()
print(X.shape)
print(Y.shape)


model.fit(X, Y, epochs=1000, shuffle=False, batch_size=30, validation_split=0.05 
        # callbacks=[#EarlyStopping(monitor='val_loss', restore_best_weights=True),
        #             TensorBoard(batch_size=100)
        #             ]
        )
model.summary()
end = timer()
print("without GPU:", end-start)
kl.predictions()
model.save("model.h5", include_optimizer=False) #model,'savedmodel')
kl.save__model()
