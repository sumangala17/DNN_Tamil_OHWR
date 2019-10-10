# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:53:50 2019

@author: Sumangala
"""

# This module trains the model according to the lSTM encoder-decoder architecture
# It uses conv2Unicode for retrieving the final dataset


#%% imports and constants
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Embedding, RepeatVector
from sklearn.model_selection import train_test_split

numclassin=70
numclassout=70
maxinlen=28
maxoutlen=28
    
encoder_input_layer = Input(shape = (seq_len, ))
decoder_input_layer = Input(shape = (seq_len, ))
    
#%% define model
    
def define_model(invocab = 20, outvocab = 25, intimesteps, outtimesteps, units):
    model = Sequential()
    model.add(Embedding(invocab,units,input_len = intimesteps, mask_zero = True))
    model.add(LSTM(units))
    model.add(RepeatVector(outtimesteps))
    model.add(LSTM(units,return_sequences = True))
    model.add(Dense(outvocab,activation = 'softmax'))
    
    return model


#%% define model

import conv2Unicode as cu
    
data = cu.getData()

train, test = train_test_split(data, test_size = 0.05, random_state = 12)

model = define_model(numclassin, numclassout, maxinlen, maxoutlen, 256)  #here numclass in = out, since no translation

rms = optimizers.RMSprop(lr = 0.001)
model.compile(optimizer = rms, loss = 'sparse_categorical_crossentropy')

filename = 'bestmodel'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#%% train model

history = model.fit(train_x, train_y.reshape(train_y.shape[0], train_y.shape[1],1), epochs=30,
                            batch_size=256,validation_split=0.2, callbacks=[checkpoint], verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()

#%% test model

model = load_model(filename)
preds = model.predict_classes(test_x.reshape((test_x.shape[0], test_x.shape[1])))











