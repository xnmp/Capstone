# retrieve data
import io, requests, os
import numpy as np, pandas as pd
import datetime as dtm
from itertools import product
from os import walk
from aux import *
from bisect import bisect_left

# -----------
# AGGREGATE THE DATA
# -----------

def aggalldata(target=None):
    walkfiles = sorted(walk('Datasets/'))
    for i, (dirpath, dirnames, filenames) in enumerate(walkfiles):
        if dirnames != []:
            for dirn in sorted(dirnames):
                print dirn.split('/')[-1]
                aggdata(dirn.split('/')[-1], target=target)
            break

def aggdata(cdate, target=None, expo=True):

    #   read all the data from the specified date, and the date before and after it
    hh = pd.DataFrame()
    folders = [f for f in sorted(os.listdir('Datasets/')) if '.csv' not in f]
    order = folders.index(cdate)
    firstday, lastday = False, False
    for i in [order-1,order,order+1]:
        # first and last days
        if i >= len(folders):
            lastday = True
            break
        if i < 0:
            firstday = True
            continue
    
        ddate = folders[i]
        for fi in [f for f in os.listdir('Datasets/' + ddate) if f[-3:] == 'csv' and 'all' not in f]:
            # if target is not none, keep only the target
            if target is not None and target not in fi:
                continue
            fpath = 'Datasets/{}/{}'.format(ddate, fi)
            fd = pd.read_csv(fpath)
            # some initial cleaning
            fd['ticker'] = fpath.split('/')[2].partition('_')[0]
            if 'pos' in fpath:
                fd = fd[fd['bid_price'] < fd['offer_price']]
                fd['type'] = False
            fd['time'] = fd['time'].apply(lambda x: dtm.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            # add random dummy trades
            if 'trade' in fi:
                fd = dummytrades(fd)
            hh = pd.concat([hh, fd], ignore_index=True)
    
    # create a flag for trade or order
    hh['type'] = hh['type'].astype(np.bool)
    hh.loc[np.isnan(hh['type']),['type']] = True
    
    # normalize times, omit the date
    hh = hh.rename(columns={'time':'datetime'})
    firsttrade = dtm.time(15,30,0)
    # firsttrade = hh[(hh['type']) & (hh['trade_qty']!=0)]['datetime'].min().time()
    hh['date'] = hh['datetime'].apply(lambda x: x.date())
    hh['time'] = hh['datetime'].apply(lambda x: subtimes(x.time(), firsttrade).total_seconds() / 234)
    
    hh = cleandata(hh, target=target, firstday=firstday, lastday=lastday, cdate=cdate)
    
    # export
    if expo:
        if target is None:
            hh.to_csv('Datasets/' + cdate + '/all.csv', index=False)
        else:
            hh.to_csv('Datasets/{}/all_{}.csv'.format(cdate, target), index=False)
    
    return hh

def cleandata(hh, target=None, firstday=False, lastday=False, cdate='1004'):

    # integrate trades and orders, fillna
    hh = hh.sort_values(['ticker','datetime','type'], ascending=[0,1,0])
    hh[['bid_price', 'bid_qty', 'offer_price', 'offer_qty']] = hh[['bid_price', 'bid_qty', 'offer_price', 'offer_qty']].fillna(method='ffill')
    hh['trade_price'] = hh['trade_price'].fillna((hh['bid_price'] + hh['offer_price'])/2)
    hh['trade_qty'] = hh['trade_qty'].fillna(0)
    
    # inverse weighted midpoint
    hh['inv_weighted_pred'] = hh['bid_price'] * hh['offer_qty'] / (hh['offer_qty'] + hh['bid_qty']) + hh['offer_price'] * hh['bid_qty'] / (hh['offer_qty'] + hh['bid_qty'])
    
    # if there are orders appearing in the same second as real trades, remove the orders
    hh['truetrade'] = hh['type'] & (hh['trade_qty']!=0)
    nt = pd.DataFrame(hh.groupby('datetime').truetrade.nunique().rename('counttypes'))
    hh = hh.merge(nt, left_on='datetime', right_index=True)
    hh = hh[(hh['counttypes']!=2) | hh['type']]
    hh = hh.drop(['counttypes','truetrade'], axis=1)
    
    # days until next: 1 usually, 3 for fridays, 4 for long weekends etc
    nextdate = hh['datetime'].iloc[-1].date()
    f = lambda x: (nextdate - x.date()).total_seconds() / 86400
    daysnext = hh['datetime'].apply(f)
    
    # time until market open
    hh['timeopen'] = np.where((hh['time'] < 0), -hh['time'], 0)
    hh['timeopen'] = np.where((hh['time'] > 100), 369.23 * daysnext - hh['time'], hh['timeopen'])
    if lastday:
        hh[hh['timeopen'] > 100] = 0
        
    # time since the last observation
    hh = hh.sort_values(['date','time'])
    hh['time_since_last'] = (hh['datetime'] - hh['datetime'].shift()).apply(lambda x: x.total_seconds() / 234)
    
    # fill in the first "time since last"
    if firstday:
        hh.loc[hh.index[0],'time_since_last'] = hh['time'].iloc[0] - 102 + 86400 / 234
    
    # add predictions
    hh = addtargets(hh)
    
    # just keep the dates that are of concern
    hh = hh[hh['date'] == dtm.date(2016, int(cdate[:2]), int(cdate[2:]))]
    
    hh = hh.reset_index(drop=True)
    
    return hh

def addtargets(df, target='JPM', ticksize=0.01, lookahead=0, type1='trade', type2='price'):
    # type1 = trade, bid, offer
    # type2 = price, qty, time
    # lookahead is the amount of time in seconds to wait before checking for the next trade
    dft = (df['ticker'] == target)
    pred = 'next_{}_{}'.format(type1, type2)
    
    # midpoint
    df.loc[dft,'midpoint'] = (df.loc[dft,'bid_price'] + df.loc[dft,'offer_price']) / 2
    df.loc[:,'midpoint'] = df['midpoint'].ffill()
    
    # transform to midpoint and spread for orders, distance from midpoint for trades
    df['midpoint_change'] = df['midpoint'] - df['midpoint'].shift()
    df['spread'] = df['offer_price'] - df['bid_price']
    df['oddeven'] = (df['spread'] / ticksize).apply(round) % 2
    # df['spread_change'] = df['spread'] - df['spread'].shift()
    df['trade_price_d'] = df['trade_price'] - df['midpoint']
    # normalize quantities
    df['bid_qty'] = df['bid_qty'] / 1000
    df['offer_qty'] = df['offer_qty'] / 1000
    df['trade_qty'] = df['trade_qty'] / 1000
    
#---ADD THE PREDICTION TARGETS
    # real trades are where the prediction trade_qty is not zero
    if type1 == 'trade':
        nextpredcond = (df[dft]['trade_qty'].shift(-1) != 0)
    # real order book updates are different from their previous ones
    else:
        nextpredcond = ((df[dft]['bid_qty'] != df[dft]['bid_qty'].shift(-1)) & (df[dft]['offer_qty'] != df[dft]['offer_qty'].shift(-1)))
    nextpred = df[dft][type1 + '_' + type2].shift(-1)
    
    #   fill it in
    df.loc[dft,pred] = np.where(nextpredcond, nextpred, np.nan)
    df.loc[:,pred] = df[pred].bfill()
    df['target'] = (df[pred] - df['midpoint']).apply(lambda x: round2(x, ticksize / 2))
    
    # clean out observations with no target
    # df = df[~np.isnan(df['target'])]
    
    # import matplotlib.pyplot as plt
    # bins = np.linspace(-0.04, 0.04, 18)
    # plt.hist(df[df['oddeven']==1]['target'].dropna(), bins)
    # plt.show()
    
    return df

# ---------------------------
# PREPARE AGGREGATED DATA FOR MODEL
# ---------------------------

def generateinput(
    target='JPM', idate='1004', ldate=None, ticksize=0.01, crow=0, 
    pricelevels=51,
    window=32, # how far to unroll, how far we BPTT
    numrows=512, # how many timesteps from the original dataframe we need
    step=1, # how many timesteps each successive input is
        # for example if window=3 and step=2 then [1,2,3,4,5] gets unrolled into [1,2,3], [3,4,5]
    outside=0, # whether to include observations outside of market hours
    model=None, # have access to the model from inside the generator
    reagg=False,
    verbose=True,
    cols=['time','midpoint_change','spread','oddeven',
            'trade_price_d','bid_qty','offer_qty','trade_qty',
            'type','timeopen','time_since_last']):
    
    cdate = idate
    # cfile = preparedata(cdate, outside=outside, reagg=reagg)
    newfile = True
    runcount = 0
    # build the dataset
    while True:
        # we first get 10% more than we need, and throw away the extras later
        if newfile:
            cfile = preparedata(cdate, outside=outside, reagg=reagg)
            if verbose:
                print 'GenerateInput: Opened new file:{}, size: {}'.format(cdate, len(cfile))
            newfile = False
        
        # print 'crow:', crow, 'cdate:', cdate
        rowuntil = crow + numrows + window - 1
        df = cfile.iloc[crow:rowuntil, :].copy()
        crow += numrows
        l = len(cfile)
        # if the current dataset doesn't contain enough data
        if rowuntil >= l:
            rowsleft = rowuntil - l
            # if verbose:
            #     print 'End of file, crow:', crow, 'cdate:', cdate
            cdate = incrementdate(cdate, ldate=ldate, idate=idate)
            newfile = True # sets a new file to be loaded on the next batch
            # reset the state when we return to the beginning
            if cdate == idate:
                if verbose:
                    print 'GenerateInput: reached the end, resetting states'
                # reset the states
                if model is not None:
                    model.reset_states()
                    # save the model weights
                    # model.save_weights('model_weights{}.h5'.format(runcount))
            
            # cfile = pd.concat([cfile.iloc[crow:], preparedata(cdate, outside=outside)])
            # cfile = preparedata(cdate, outside=outside, reagg=reagg)
            newstart = max(l - crow, 0)
            crow = 0        
            
            # pad the dataframe
            paddf = np.empty((rowsleft, len(df.columns)))
            paddf[:] = np.nan
            df = pd.concat([df, pd.DataFrame(paddf, columns=df.columns)])
            df['trade_qty'] = df['trade_qty'].fillna(0)
            df['midpoint_change'] = df['midpoint_change'].fillna(0)
            df['time_since_last'] = df['time_since_last'].fillna(0)
            df['timeopen'] = df['timeopen'].fillna(239.23)
            df['type'] = df['type'].fillna(False).astype(np.bool)
            df = df.fillna(method='ffill')
            
            # append part of the next dataset (not in use)
            # print crow, l, rowuntil, rowsleft, newstart
            # df = pd.concat([df, cfile.iloc[newstart: newstart + rowsleft]])

        # change the midpoint for files between files
        f = df.loc[df.index[0],'midpoint_change']
        df['midpoint_change'] = df['midpoint'] - df['midpoint'].shift()
        df.loc[df.index[0],'midpoint_change'] = f

        # trim the dataset
        df = df.iloc[:numrows + window - 1]
        
        # convert to numpy array
        targets = np.array(df['target'])
        df = np.array(df[cols])
        runcount += 1
        
        # THE DATA: note that we use zeros to set aside the memory at the outset
        X = np.zeros((numrows / step, window, len(df[0])))
        for i in range(0, numrows / step):
            X[i] = df[i*step:i*step+window, :]
        # puts the targets into bins
        y = makeprob(targets, numrows / step, pricelevels, ticksize, step, window)
        yield X, y

def getdummies(df):
    df = pd.concat([df, pd.get_dummies(df['ticker']).astype(np.bool)], axis=1)
    for ticker in tickers:
        if ticker not in df.columns:
            df[ticker] = False
    # sort the columns alphabetically
    return df.reindex_axis(sorted(df.columns), axis=1)

def preparedata(idate, ldate=None, tickers = [u'BAC', u'BBD', u'BBT', u'BK', u'BNS', u'BOH', u'BXS', u'C', u'CBU',
    u'CFR', u'CMA', u'COF', u'CPF', u'DB', u'FBP', u'FCF', u'FSB', u'HTH',
    u'ITUB', u'JPM', u'KEY', u'MFG', u'MTB', u'OFG', u'PB', u'PNC', u'RF',
    u'SCNB', u'SHG', u'SNV', u'STI', u'STL', u'STT', u'TCB', u'UBS', u'USB',
    u'VLY', u'WFC'],
    outside=0,
    target='JPM',
    reagg=False):
    
    if ldate is None:
        ldate = idate
    
    df = pd.DataFrame()
    for d in datelist2(idate, ldate):
        if reagg:
            hh = aggdata(d, target=target, expo=False)
        else:
            if target is None:
                hh = pd.read_csv('Datasets/{}/all.csv'.format(d))
            else:
                hh = pd.read_csv('Datasets/{}/all_{}.csv'.format(d, target))
        df = pd.concat([df, hh])
    
    # add the dummies for the ticker
    # df = getdummies(df)
    
    # remove everything outside of 4 minutes of market open and close
    df = df[(df['time'] <= 100 * (1 + outside)) & (df['time'] >= (-outside))]
    df['time_since_last'].iloc[0] = 267.23
    
    # reset index
    # df = df.reset_index(drop=True)
    
    return df
