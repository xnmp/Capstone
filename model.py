import numpy as np, pandas as pd
np.random.seed = 7

from processing import generateinput
from keras.models import Sequential, load_model, model_from_json
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

def makemodel(
    batchsize=0, window=0, numcols=0, pricelevels=0, dropout=0, 
    savedweights=None, savedmodel=None):

    if savedmodel is not None or savedweights is not None:
        json_file = open('model.json','r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        json_file.close()
        model.load_weights(savedweights)
        # model = load_model(savedmodel)
        model.reset_states()
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print 'Loaded model from disk'
        return model

    print 'Loading model...'
    model = Sequential()
    model.add(LSTM(512,
        return_sequences=True,
        batch_input_shape=(batchsize, window, numcols),
        stateful=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=True, stateful=True))
    model.add(Dropout(dropout))
    # model.add(LSTM(256, return_sequences=True, stateful=True))
    # model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=False, stateful=True))
    model.add(Dropout(dropout))
    model.add(Dense(pricelevels))
    model.add(Activation('softmax'))

    # load the weights
    if not savedweights is None:
        model.load_weights(savedweights)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print '--DONE'
    return model


# class ResetStatesCallback(Callback):
#     def __init__(self):
#         self.counter = 0

#     def on_batch_begin(self, batch, logs={}):
#         pass
        # batch is the batch number
        # print batch
        # if self.counter % max_len == 0:
        #     self.model.reset_states()
        # self.counter += 1
