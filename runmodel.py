import numpy as np, pandas as pd
np.random.seed = 7
# sudo ldconfig /usr/local/cuda-8.0/lib64
from processing import generateinput, preparedata
from model import makemodel
from keras.callbacks import EarlyStopping, ModelCheckpoint

# numcols 48 or 49
numcols, pricelevels, window, step = 11, 31, 32, 1
epochsize = 131072 / 4
epochs = 200
batchsize = 64 # use a smaller batch size
dropout = 0.2

model = makemodel(savedweights='model_weights.h5')


# -- TRAIN THE MODEL

# # model = makemodel(
# #     batchsize=batchsize,
# #     window=window, numcols=numcols, pricelevels=pricelevels,
# #     dropout=dropout,
# #     )

gen = generateinput(
    numrows=batchsize*step, window=window, pricelevels=pricelevels, step=step,
    initialdate='1004',
    model=model)

# -- validation
# checkpoint = ModelCheckpoint(
#     filepath="model_weights.h5", verbose=1,
#     # save_best_only=True,
#     # save_weights_only=False
#     )
# es = EarlyStopping(
#     monitor='val_loss',
#     # min_delta=0.08,
#     patience=10)
# valgen = generateinput(
#     numrows=batchsize*step, pricelevels=pricelevels, window=window, step=step,
#     initialdate='1102')

hist = model.fit_generator(gen, 
    samples_per_epoch=epochsize, nb_epoch=epochs,
    # ========
    # validation_data=valgen,
    # nb_val_samples=32,
    # callbacks=[],
    # ========
    verbose=2,
    max_q_size=10)

# -- training history
pd.DataFrame(hist.history).to_csv('train_history.csv')

# -- save the model
model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
model.save_weights('model_weights.h5')

# -- DO THE PREDICTIONS
numpreds, cdate = 32000, '1004'
predgen = generateinput(
    numrows=batchsize*step, window=window, pricelevels=pricelevels, step=step, 
    initialdate=cdate)
preds = model.predict_generator(predgen, val_samples=numpreds)
preds = pd.concat([preparedata(cdate)[['time','midpoint_change','spread','oddeven',
            'trade_price_d','bid_qty','offer_qty','trade_qty',
            'type','timeopen','time_since_last']].iloc[:numpreds].reset_index(drop=True)
            , pd.DataFrame(preds)
            ], axis=1)

preds.to_csv('preds.csv')
