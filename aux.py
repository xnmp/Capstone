# retrieve data
import io, requests, os
import numpy as np, pandas as pd
import datetime as dtm
from itertools import product
from os import walk, path
from bisect import bisect_left
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mabs

pricelevels, window, batchsize, step, numcols  = 31, 32, 64, 1, 11

def round2(number,roundto):
    return (round(number / roundto) * roundto)

def subtimes(t1,t2):
    # subtract two times (not datetimes)
    return dtm.datetime.combine(dtm.date.min, t1) - dtm.datetime.combine(dtm.date.min, t2)

def timefromhrs(x):
    minutes = x % 1 * 60.0
    seconds = minutes % 1 * 60
    return dtm.time(int(x // 1), int(minutes // 1), int(seconds // 1))

def incrementdate(d, delta=1, idate='1004', ldate=None, shuffle=False): #d is a string
    # newdate = date(2016, int(d[:2]), int(d[2:])) + timedelta(days=delta)
    # return str.zfill(str(newdate.month), 2) + str.zfill(str(newdate.day), 2)
    dates = sorted([fi for fi in os.listdir('Datasets') if '.csv' not in fi])
    # can shuffle the dates list if needed here
    # would need to turn it into a generator
    datedict = {dates[i]:dates[i+1] for i in range(len(dates)-1)}
    if d == ldate or d not in datedict:
        return idate
    else:
        return datedict[d]

def takeClosest(myList, myNumber):
    # Assumes myList is sorted. Returns closest value to myNumber.
    # If two numbers are equally close, return the smallest number.
    # Credit goes to http://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def genbins(pricelevels, ticksize, exponent=0.35):
    #   this is how we do it:
    #   40% of the bins are one halftick
    #   24% are two halfticks, 15% are 3, 9% are 4, 6% are 5, it's about *3/5 each time
    lleft, htick = (pricelevels - 1) / 2, ticksize / 2
    temp, cnum = lleft, htick
    multiplier, i, j = 1, 1, 1
    yield 0
    while i <= temp:
        if j > lleft * exponent:
            lleft -= j
            multiplier += 1
            j = 0
        yield cnum
        yield -cnum
        cnum += htick * multiplier
        j += 1
        i += 1

def makeprob(targets, inputlen, pricelevels, ticksize, step, window, tsteps=1):
    # turn a list of numbers into categorical
    bins = sorted(list(genbins(pricelevels, ticksize)))
    bindict = {x:i for i, x in enumerate(bins)}
    htick = ticksize / 2
    
    if tsteps > 1:
        raise NotImplementedError
        
    y = np.zeros((inputlen, pricelevels))
    for i in range(0, inputlen):
        # if the window is 32 the first target is at row index 31
        y[i, bindict[takeClosest(bins, targets[i*step+window-1])]] = 1
    return y

def dummytrades(df, tradeprop=0.1, vari=2):
    numtrades = len(df)
    ftrades = pd.DataFrame()
    #   create a dummy dataframe
    firsttrade = df['time'].min().hour + 0.5
    lasttrade = round(df['time'].max().hour)
    dist = np.random.uniform(firsttrade, lasttrade, int(tradeprop*numtrades))
    ftrades['time'] = [dtm.datetime.combine(df['time'].iloc[0].date(), timefromhrs(x)) for x in dist]
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
    df['type'] = True
    return df

def datelist(idate, ldate):
    iterdates = []
    idate2 = idate
    while idate2 <= ldate:
        iterdates.append(dtm.date.strftime(idate2, '%m%d'))
        idate2 += dtm.timedelta(days=1)
    return iterdates

def datelist2(idate, ldate):
    iterdates = []
    idate = dtm.datetime.strptime(idate, '%m%d')
    ldate = dtm.datetime.strptime(ldate, '%m%d')
    idate2 = idate
    while idate2 <= ldate:
        tempd = dtm.date.strftime(idate2, '%m%d')
        if path.isdir('Datasets/{}'.format(tempd)):
            iterdates.append(tempd)
        idate2 += dtm.timedelta(days=1)
    return iterdates

def makevaldata(idate='1114', ldate=None, target='JPM', outside=0):
    
    idate = dtm.date(year=2016, month=int(idate[:2]), day=int(idate[2:]))
    if ldate == None:
        ldate = dtm.date.today() - dtm.timedelta(days=1)
    else:
        ldate = dtm.date(year=2016, month=int(ldate[:2]), day=int(ldate[2:]))
    
    df = pd.DataFrame()
    for date in datelist(idate, ldate):
        try:
            hh = pd.read_csv('Datasets/{}/all_{}.csv'.format(date, target))
        except:
            continue
        df = pd.concat([df, hh])
    df = df[(df['time'] <= 100 * (1 + outside)) & (df['time'] >= (-outside))]
    df.to_csv('Datasets/validation_data.csv')
    
    return df
    
def epochlength(idate='1004', ldate=None, target='JPM', outside=0, val=False):
    
    idate = dtm.date(year=2016, month=int(idate[:2]), day=int(idate[2:]))
    if ldate == None:
        ldate = dtm.date.today()
    else:
        ldate = dtm.date(year=2016, month=int(ldate[:2]), day=int(ldate[2:]))
    
    res = 0
    for date in datelist(idate, ldate):
        try:
            hh = pd.read_csv('Datasets/{}/all_{}.csv'.format(date, target))
        except:
            continue
        hh = hh[(hh['time'] <= 100 * (1 + outside)) & (hh['time'] >= (-outside))]
        # need x such that x % 64 == 0 and x + 95 > len(hh)
        l = len(hh) - 31
        # first multiple of 64 >= l
        if l % 64 == 0:
            res += l
        else:
            res += l / 64 * 64 + 64
        if val:
            res -= 64
        
    return res

binlist = sorted(list(genbins(31, 0.01)))
bindict = {i:x for i, x in enumerate(binlist)}

def stdfromdist(s):
    std = 0
    for i in range(31):
        val = bindict[i]
        std += s[str(i)] * (val - s['model_d_pred']) ** 2
    std = np.sqrt(std)
    return std

# combine these into one later
def momentum(ii, df, var='target'):
    # credit goes to stackoverflow question 21040766
    x_df = df.iloc[map(int, ii)]
    x_df = x_df[x_df['trade_qty']>0]
    return pearsonr(x_df['trade_price_d'], x_df[var]-x_df['trade_price_d'])[0]

def rollpearson(ii, df, var='model_d_pred', func=pearsonr):
    # credit goes to stackoverflow question 21040766
    x_df = df.iloc[map(int, ii)]
    x_df = x_df[x_df['trade_qty']>0]
    return func(x_df[var], x_df['target'])[0]

def rollmse(ii, df, var='model_d_pred', func=mse):
    # credit goes to stackoverflow question 21040766
    x_df = df.iloc[map(int, ii)]
    x_df = x_df[x_df['trade_qty']>0]
    return func(x_df[var], x_df['target'])

def sumprobs(row):
    ticksize = 0.01
    return sum([row[str(i)] * bindict[i] for i in range(pricelevels)])

def sumstds(stdevs, window):
    return np.sqrt(np.power(stdevs, 2).sum() / window)