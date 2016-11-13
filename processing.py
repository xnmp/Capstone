# retrieve data
import io, requests, os
import numpy as np, pandas as pd
from datetime import datetime as dt, timedelta, time, date
from itertools import product
from os import walk
from aux import *
from bisect import bisect_left

# -----------
# AGGREGATE THE DATA
# -----------

def readdata(fpath):
    # this contains raw data, plus the ticker
    hh = pd.read_csv(fpath)
    hh['ticker'] = fpath.split('/')[2].partition('_')[0]
    hh['time'] = hh['time'].apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return hh

def dummytrades(df, tradeprop=0.1, vari=2):
    numtrades = len(df)
    ftrades = pd.DataFrame()
    #   create a dummy dataframe
    firsttrade = df['time'].min().hour + 0.5
    lasttrade = round(df['time'].max().hour)
    ftrades['time'] = [dt.combine(df['time'].iloc[0].date(), timefromhrs(x)) for x in np.random.uniform(firsttrade,lasttrade,int(tradeprop*numtrades))]
    ftrades['trade_qty'] = 0
    #   append and sort
    df = df.append(ftrades, ignore_index=True)
    df.sort_values('time', inplace=True)
    #   set a standard deviation
    df['std'] = df['trade_price'].rolling(window=10, min_periods=1).apply(np.nanstd)
    df.loc[df['std'] == 0, 'std'] = 10**(-9)
    #   sample from random distribution
    df['randprice'] = df['std'].apply(lambda x: np.random.normal(0, vari * x))
    df.loc[~np.isnan(df['trade_price']), 'randprice'] = 0
    #   add it to the previously traded price
    df.fillna(method='ffill', inplace=True)
    df['trade_price'] = df['randprice'] + df['trade_price']
    df.drop(['randprice','std'],axis=1,inplace=True)
    #   add a flag so that we know it's a trade
    df['type'] = 1
    return df

def aggalldata(target=None):
    walkfiles = sorted(walk('Datasets/'))
    for i, (dirpath, dirnames, filenames) in enumerate(walkfiles):
        if dirnames != []:
            for dirn in sorted(dirnames):
                print dirn.split('/')[-1]
                aggdata(dirn.split('/')[-1], target=target)
            break

def aggdata(cdate, target=None):

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
            fd = readdata(fpath)
    #       add random dummy trades
            if 'trade' in fi:
                fd = dummytrades(fd)
            hh = pd.concat([hh, fd], ignore_index=True)

    # integrate trades and orders, fillna
    hh = hh.sort_values(['ticker','time','type'])
    hh[['bid_price', 'bid_qty', 'offer_price', 'offer_qty']] = hh[['bid_price', 'bid_qty', 'offer_price', 'offer_qty']].fillna(method='ffill')
    hh['trade_price'] = hh['trade_price'].fillna((hh['bid_price'] + hh['offer_price'])/2)
    hh['trade_qty'] = hh['trade_qty'].fillna(0)

    # create a flag for trade or order
    hh.loc[np.isnan(hh['type']),['type']] = False
    hh['type'] = hh['type'].astype(np.bool)

    # if there are orders appearing in the same second as trades, remove the orders
    hh['truetrade'] = hh['type'] & (hh['trade_qty']!=0)
    nt = pd.DataFrame(hh.groupby('time').truetrade.nunique().rename('counttypes'))
    hh = hh.merge(nt, left_on='time', right_index=True)
    hh = hh[(hh['counttypes']!=2) | (hh['truetrade'])]
    hh = hh.drop(['counttypes','truetrade'], axis=1)

    # inverse weighted midpoint
    hh['inv_weighted_midpoint'] = hh['bid_price'] * hh['offer_qty'] + hh['offer_price'] * hh['bid_qty']
    hh['inv_weighted_d'] = hh['inv_weighted_midpoint'] - hh['midpoint']

    # days until next: 1 usually, 3 for fridays, 4 for long weekends etc
    hh = hh.rename(columns={'time':'datetime'})
    nextdate = hh['datetime'].iloc[-1].date()
    f = lambda x: (nextdate - x.date()).total_seconds() / 86400
    daysnext = hh['datetime'].apply(f)

    # normalize times, omit the date
    firsttrade = hh[(hh['type']) & (hh['trade_qty']!=0)]['datetime'].min().time()
    hh['date'] = hh['datetime'].apply(lambda x: x.date())
    hh['time'] = hh['datetime'].apply(lambda x: subtimes(x.time(), firsttrade).total_seconds() / 234)
    
    # time until market open
    hh['timeopen'] = np.where((hh['time'] < 0), -hh['time'], 0)
    hh['timeopen'] = np.where((hh['time'] > 100), 369.23 * daysnext - hh['time'], hh['timeopen'])
    if lastday:
        hh[hh['timeopen'] > 100] = 0

    # time since the last observation
    hh = hh.sort_values(['date','time'])
    hh['time_since_last'] = (hh['datetime'] - hh['datetime'].shift()).apply(lambda x: x.total_seconds() / 234)

    # add predictions
    hh = addtargets(hh)

    # just keep the dates that are of concern
    hh = hh[hh['date'] == date(2016, int(cdate[:2]), int(cdate[2:]))]
    
    # fill in the first "time since last"
    if firstday:
        hh.loc[hh.index[0],'time_since_last'] = hh['time'].iloc[0] - 102 + 86400 / 234

    # export
    if target is None:
        hh.to_csv('Datasets/' + cdate + '/all.csv', index=False)
    else:
        hh.to_csv('Datasets/{}/all_{}.csv'.format(cdate, target), index=False)

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

    # import matplotlib.pyplot as plt
    # bins = np.linspace(-0.04, 0.04, 18)
    # plt.hist(df[df['oddeven']==1]['target'].dropna(), bins)
    # plt.show()
    return df

# ---------------------------
# PREPARE AGGREGATED DATA FOR MODEL
# ---------------------------

def generateinput(
    target='JPM', initialdate='1004', 
    pricelevels=51,
    ticksize=0.01,
    window=32, # how far to unroll, how far we BPTT
    numrows=512, # how many timesteps from the original dataframe we need
    step=1, # how many timesteps each successive input is
        # for example if window=3 and step=2 then [1,2,3,4,5] gets unrolled into [1,2,3], [3,4,5]
    outside=0, # whether to include observations outside of market hours
    crow=0,
    model=None, # have access to the model from inside the generator
    cols=['time','midpoint_change','spread','oddeven',
            'trade_price_d','bid_qty','offer_qty','trade_qty',
            'type','timeopen','time_since_last']
    ):

    cdate = initialdate
    cfile = preparedata(cdate, outside=outside)
    # build the dataset
    while True:
        # we first get 10% more than we need, and throw away the extras later
        rowuntil = crow + numrows + window
        df = cfile.iloc[crow:rowuntil, :].copy()
        crow += numrows
        l = len(cfile)
        # if the current dataset doesn't contain enough data
        if rowuntil > l:
            rowsleft = rowuntil - l
            cdate = incrementdate(cdate)
            # cfile = pd.concat([cfile.iloc[crow:], preparedata(cdate, outside=outside)])
            cfile = preparedata(cdate, outside=outside)
            newstart = max(l - crow, 0)
            crow = 0

            # reset the state when we return to the beginning
            if cdate == '1004':
                print 'GenerateInput: Restarting from the beginning'
                # reset the states
                if model is not None:
                    model.reset_states()
                    print model.summary()

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
            
            # append part of the next dataset
            # print crow, l, rowuntil, rowsleft, newstart
            # df = pd.concat([df, cfile.iloc[newstart: newstart + rowsleft]])

        # change the midpoint for files between files
        f = df.loc[df.index[0],'midpoint_change']
        df['midpoint_change'] = df['midpoint'] - df['midpoint'].shift()
        df.loc[df.index[0],'midpoint_change'] = f

        # trim the dataset
        df = df.iloc[:numrows + window]

        # convert to numpy array
        targets = np.array(df['target'])
        df = np.array(df[cols])
        # drop(['next_trade_price','bid_price','offer_price'], axis=1)

        # THE DATA: note that we use zeros to set aside the memory at the outset
        X = np.zeros((numrows / step, window, len(df[0])))
        for i in range(0, numrows / step):
            X[i] = df[i*step:i*step + window, :]
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

def preparedata(cdate, tickers = [u'BAC', u'BBD', u'BBT', u'BK', u'BNS', u'BOH', u'BXS', u'C', u'CBU',
    u'CFR', u'CMA', u'COF', u'CPF', u'DB', u'FBP', u'FCF', u'FSB', u'HTH',
    u'ITUB', u'JPM', u'KEY', u'MFG', u'MTB', u'OFG', u'PB', u'PNC', u'RF',
    u'SCNB', u'SHG', u'SNV', u'STI', u'STL', u'STT', u'TCB', u'UBS', u'USB',
    u'VLY', u'WFC'],
    outside=0,
    target='JPM'):
    
    if target is None:
        df = pd.read_csv('Datasets/{}/all.csv'.format(cdate))
    else:
        df = pd.read_csv('Datasets/{}/all_{}.csv'.format(cdate, target))

    # add the dummy variables
    # df = getdummies(df)

    # remove everything outside of 4 minutes of market open and close
    df = df[(df['time'] <= 100 * (1 + outside)) & (df['time'] >= (-outside))]
    df['time_since_last'].iloc[0] = 267.23

    # clean out observations with no target
    df = df[~np.isnan(df['target'])]

    # reset index
    df = df.reset_index(drop=True)
    print 'GenerateInput: Opening new file:{}, size: {}'.format(cdate, len(df))
    return df
