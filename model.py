import numpy as np, pandas as pd
np.random.seed = 7

from processing import generateinput
from keras.models import Sequential, load_model, model_from_json
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback

def makemodel(
    batchsize=0, window=0, numcols=0, pricelevels=0, dropout=0, 
    savedweights=None, savedmodel=None):

    # default is 0.001 and 0.002 respectively
    myadam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    mynadam = Nadam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    if savedmodel is not None or savedweights is not None:
        json_file = open('Results/model.json','r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        json_file.close()
        model.load_weights(savedweights)
        # model = load_model(savedmodel)
        model.reset_states()
        model.compile(loss='categorical_crossentropy', optimizer=mynadam)
        print 'Loaded model from disk'
        return model

    print 'Loading model...'
    model = Sequential()
    model.add(LSTM(512,
        return_sequences=True,
        batch_input_shape=(batchsize, window, numcols),
        stateful=True))
    model.add(Dropout(dropout))
    # model.add(LSTM(256, return_sequences=True, stateful=True))
    # model.add(Dropout(dropout))
    # model.add(LSTM(512, return_sequences=True, stateful=True))
    # model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=True, stateful=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=False, stateful=True))
    model.add(Dropout(dropout))
    model.add(Dense(pricelevels))
    model.add(Activation('softmax'))

    # load the weights
    if not savedweights is None:
        model.load_weights(savedweights)
    model.compile(loss='categorical_crossentropy', optimizer=myadam)
    # save the model architecture
    model_json = model.to_json()
    with open('Results/model.json','w') as json_file:
        json_file.write(model_json)
    print '--DONE'
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.count = 1
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        pd.Series(self.losses).to_csv('Results/loss_history{}.csv'.format(self.count))
        self.losses = []
        self.count += 1

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
