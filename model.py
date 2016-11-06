import numpy as np, pandas as pd
from processing import generateinput
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------
# HYPER PARAMETERS
# ---------------

numcols, pricelevels, window, step = 48, 51, 20, 4
epochsize = 131072
epochs = 4
batchsize = 256

# -----------
# THE MODEL
# -----------

# model = load_model('model.h5')
# model.reset_states()

model = Sequential()
model.add(LSTM(32, 
    return_sequences=True, 
    batch_input_shape=(batchsize, window, numcols), 
    stateful=True))
model.add(Dropout(0.20))
# model.add(LSTM(512, return_sequences=True, stateful=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256, return_sequences=True, stateful=True))
# model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False, stateful=True))
model.add(Dropout(0.20))
model.add(Dense(pricelevels))
model.add(Activation('softmax'))
# model.load_weights('weights.best.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

filepath = "weights.best.hdf5" # "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=2)
gen = generateinput(numrows=batchsize*4)
valgen = generateinput(initialdate='1103', numrows=batchsize*4)

# for i in range(epochs):
#     print 'Epoch', i
model.fit_generator(gen, 
    samples_per_epoch=epochsize, 
    nb_epoch=epochs,
    validation_data=valgen,
    nb_val_samples=32,
    callbacks=[checkpoint, es],
    )

# model.reset_states()
# model.save('model.h5')
