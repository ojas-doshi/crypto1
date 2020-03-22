from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Reshape
from keras.optimizers import Adam, RMSprop
from numpy import genfromtxt
import numpy as np
from timeit import default_timer as timer 
# import tensorflow

dataset = genfromtxt('LTC1.csv', delimiter=',')

X = dataset[:,0:30]
Y = dataset[:,30:]

print("X",X.shape)
print("Y",Y.shape)
start = timer() 

# X=[]
# Y=[]

# for i in range(len(x)):
#     if not -99999999 in x[i]:
#         X.append(np.array([x[i]]))
#         Y.append(np.array(y[i]))

# X=np.array(X)
# Y=np.array(Y)

# print(dataset)
# x = dataset[:,0:12]
# y = dataset[:,23:]
# dataset1 = dataset.drop([13,14,15,16,17,18,19,20,21,22],axis=1)



model = Sequential()

# # Embedding layer
# model.add(
#     Embedding(input_dim=3000,
#               input_length = 28,
#               output_dim=28,
#               weights=[dataset],
#               trainable=False,
#               mask_zero=True))

# Masking layer for pre-trained embeddings
# model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(Reshape((1,30)))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128))

# Fully connected layer

model.add(Dense(25, activation='relu'))


# Dropout for regularization
# model.add(Dropout(0.5))

# Output layer
model.add(Dense(3, activation='softmax'))

# Compile the model
# optimizer1=Adam(lr=.1, decay=0.0001)
optimizer1=RMSprop(lr=.001)#, decay=0.0001)
model.compile(loss='categorical_crossentropy', optimizer = optimizer1, metrics=['accuracy'])

print(X.shape)
print(Y.shape)

model.fit(X,Y, epochs=100000, shuffle=True,batch_size=2048, validation_split=0.05, validation_freq=50)
# model.fit(X,Y, epochs=10000, shuffle=True,batch_size=10, validation_split=0.05)


predictions=model.predict(X)

print(predictions)


print("without GPU:", timer()-start)
