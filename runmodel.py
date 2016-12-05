import numpy as np, pandas as pd
np.random.seed = 7
# sudo ldconfig /usr/local/cuda-8.0/lib64
from aux import * # genbins, takeClosest, epochlength, stdfromdist, momentum
from processing import generateinput, preparedata
from model import makemodel, LossHistory
from keras.callbacks import EarlyStopping, ModelCheckpoint # ReduceLROnPlateau

def trainmodel(model, idate='1004', ldate=None, val_idate='1114', val_ldate='1118',
               batchsize=64, epochs=50, epochsize=None):
    
    # count the length of an epoch
    if epochsize is None:
        epochsize = epochlength(idate=idate, ldate=ldate) #32768
        print 'Epoch Size:', epochsize
    
    # -- callbacks
    checkpoint = ModelCheckpoint(filepath="Results/model_weights-{epoch:02d}-{val_loss:.2f}.h5", verbose=1,
        save_best_only=True, save_weights_only=True)
    es = EarlyStopping(monitor='val_loss', patience=9) # min_delta=0.08,
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                   patience=6, min_lr=0.0001)
    losshist = LossHistory()
    
    # validation data
    valgen = generateinput(
        verbose=True, model=model, numrows=batchsize*step, window=window, pricelevels=pricelevels, step=step,
        idate=val_idate, ldate=val_ldate)
    
    gen = generateinput(
        numrows=batchsize*step, window=window, pricelevels=pricelevels, step=step,
        idate=idate, ldate=ldate, model=model, 
        reagg=True)
    
    hist = model.fit_generator(gen, 
        samples_per_epoch=epochsize, nb_epoch=epochs,
        # ========
        validation_data=valgen,
        nb_val_samples=epochlength(idate=ldate, ldate=val_ldate, val=True),
        callbacks=[checkpoint, es, reduce_lr, losshist],
        # ========
        verbose=2, max_q_size=10)
    
    # -- save training history
    pd.DataFrame(hist.history).to_csv('Results/train_history.csv')
    
    # -- save the model
    model_json = model.to_json()
    with open('Results/model.json','w') as json_file:
        json_file.write(model_json)
    # model.save_weights('Results/model_weights.h5')
    return model

def modelpredict(idate='1114', ldate='1118', numpreds=None, outside=0):
    ticksize = 0.01
    cols2 = ['time','bid_price','bid_qty','offer_price','offer_qty','date',
             'trade_price','trade_qty','midpoint','spread',
             'midpoint_change','trade_price_d','next_trade_price','target','inv_weighted_pred',
             'oddeven','type','timeopen','time_since_last']
    
    if ldate is None:
        ldate = idate
    
    predsall = pd.DataFrame()
    for d in datelist2(idate, ldate):
        
        if numpreds is None:
            numpreds = epochlength(idate=d, ldate=d, val=True)
        print d, ': Number of Predictions:', numpreds
        
        predgen = generateinput(
            numrows=batchsize*step, window=window, pricelevels=pricelevels, step=step, 
            idate=d, ldate=d, outside=outside)
        
        print "Predicting..."
        preds = model.predict_generator(predgen, val_samples=numpreds)
        print "DONE"
        
        # match up with data
        hh = preparedata(d, outside=outside)[cols2]
        # need to account for the phase shift of the predictions
        firstobs = hh.index[0] + window - 1
        preds = pd.DataFrame(preds, 
                             index=range(firstobs, 
                                         firstobs+len(preds)))
        preds.columns = map(str, preds.columns)
        
        print "Merging..."
        preds = pd.merge(hh, preds,
                         left_index=True, right_index=True)
        print "DONE"
        print "1"
        # calculate the mean
        preds['model_d_pred'] = preds.apply(sumprobs, axis=1)
        preds['model_pred'] = preds['model_d_pred'] + preds['midpoint']
        preds = preds.drop(['type','timeopen','time_since_last','oddeven'], axis=1)
        print "2"
        # entropy
        # preds['closestbin'] = preds['target'].apply(lambda x: takeClosest(binlist, x))
        # preds['bin_no'] = preds['closestbin'].apply(lambda x: int(binlist.index(x)))
        # preds['entropy'] = preds.apply(lambda row: -np.log(row[str(int(row['bin_no']))]), axis=1)
        # preds = preds.drop(['closestbin','bin_no'],axis=1)
        # preds['entropy'] = np.where(preds['trade_qty']>0, preds['entropy'].shift(), 0)
        print "3"
        # standard deviation and rolling statistics
        rwind = 30
        preds['std'] = preds.apply(stdfromdist, axis=1)
        # preds['roll_std'] = preds['std'].rolling(rwind).apply(lambda x: sumstds(x, rwind))
        # preds['roll_entropy'] = preds['entropy'].rolling(rwind).mean()
        
        # momentum
        preds['ii'] = range(len(preds))
        # preds['momentum_actual'] = np.where(preds['trade_qty']>0, preds['target'] - preds['trade_price_d'], 0)
        # preds['roll_momentum_actual'] = preds.ii.rolling(rwind).apply(lambda x: momentum(x, preds)).fillna(0)
        # preds['roll_momentum_pred'] = preds.ii.rolling(rwind).apply(lambda x: momentum(x, preds, var='model_d_pred')).fillna(0)
        preds['roll_pearsonr'] = preds.ii.rolling(rwind).apply(lambda x: rollpearson(x, preds)).fillna(0)
        preds['roll_mse'] = preds.ii.rolling(rwind).apply(lambda x: rollmse(x, preds)).fillna(-1)
        print "4"
        preds['roll_pearsonr_inv'] = preds.ii.rolling(rwind).apply(lambda x: rollpearson(x, preds, var='inv_weighted_d_pred')).fillna(0)
        preds['roll_mse_inv'] = preds.ii.rolling(rwind).apply(lambda x: rollmse(x, preds, var='inv_weighted_d_pred')).fillna(-1)
        preds = preds.drop(['ii'], axis=1)
        print "5"
        preds.to_csv('Results/preds{}.csv'.format(d))
        predsall = pd.concat([predsall, preds])
    
    predsall.to_csv('Results/predsall.csv')
    return predsall

# -- load model
model = makemodel(savedweights='Results/model_weights_1.52.h5')

# -- make model from scratch
# model = makemodel(batchsize=64, window=32, numcols=11, pricelevels=31, dropout=0.2)

# -- train the model
# model = trainmodel(model, idate='1004', ldate='1111', epochs=50, epochsize=None)

# -- predictions
preds = modelpredict(idate='1115', ldate='1118')

# import matplotlib.pyplot as plt
# plt.scatter(preds['model_d_pred'], preds['target'])
# plt.show()
from scipy.stats import pearsonr
print pearsonr(preds['target'], preds['model_d_pred'])
