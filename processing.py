# retrieve data
import io, requests, os
import numpy as np, pandas as pd
from datetime import datetime as dt, timedelta, time, date
from itertools import product
from os import walk

# # rearrange folders
# import os
# for fpath in datasets:
#     fpathl = fpath.split('/')
#     ticker = fpathl[1]
#     ddate = fpathl[2].partition('_')[0]
#     dtype = fpathl[2].partition('_')[2].partition('.')[0]
#     if not os.path.exists('Datasets/' + ddate):
# 	    os.makedirs('Datasets/' + ddate)
#     os.rename(fpath, '{}/{}/{}_{}.csv'.format(fpathl[0], ddate, ticker, dtype))

def getdata(dtype, symb, date):

    url0, url1, url2, url3 = 'http://hopey.netfonds.no/', 'dump.php?date=2016', '&paper=', '.N&csv_format=csv'

    #     read the files
    url = url0 + dtype + url1 + date + url2 + symb + url3
    df = requests.get(url).content
    df = pd.read_csv(io.StringIO(df.decode('utf-8')))

    #     do some cleaning
    df['time'] = pd.to_datetime(df['time'])
    if dtype == 'pos':
        df = df.drop(['bid_depth_total','offer_depth_total'], axis=1)
        df.columns = ['time', 'bid_price', 'bid_qty', 'offer_price', 'offer_qty']
    elif dtype == 'trade':
        df = df.drop(['source','buyer','seller','initiator'], axis=1)
        df.columns = ['time', 'trade_price', 'trade_qty']

    #     stop if the dataset doesn't exist
    # maxtime = int(df['time'].apply(lambda x: x.hour).max())
    # if maxtime < 22:
    # 	# don't export if the maxtime doesn't exist
    #     return df

    #     export the file
    if not os.path.exists('Datasets/' + date):
        os.makedirs('Datasets/' + date)
        print "Made Directory", date
    df.to_csv('Datasets/' + date + '/' + symb + '_' + dtype + '.csv', index=False)

    return df

def getinddata(ind='Major Banks', currentdate=dt.date(dt.today()), window=3):

	cdate = currentdate#dt.date(2016, currentdate[:2], currentdate[2:])

	tickers = gettickers()

	sect_tickers = tickers.reset_index().groupby('Sector')['Symbol'].agg(lambda x: list(x))
	ind_tickers = tickers.reset_index().groupby('Industry')['Symbol'].agg(lambda x: list(x))

	iterdates = [str((cdate-timedelta(days=i)).month).zfill(2) + str((cdate-timedelta(days=i)).day).zfill(2) for i in range(1, window+1)]
	print iterdates
	for date in iterdates: # or tickers.index for all industries
		print "DATE:", date
		for symb, dtype in product(ind_tickers[ind], ('pos','trade')):
			if symb == 'BLW':
				continue
			print symb
			try:
			    getdata(dtype, symb, date)
			except UnicodeDecodeError:
			    print '--NOT FOUND'
			    continue

def gettickers():
    tickers = pd.read_csv('Datasets/companylist.csv')
    tickers = tickers.set_index('Symbol')[['Name','Industry','Sector','MarketCap']]
    tickers = tickers[(tickers.MarketCap > 10**8) & (tickers.Sector != 'n/a')]
    return tickers

#this contains raw data, plus the ticker
def readdata(fpath):
    hh = pd.read_csv(fpath)
    hh['ticker'] = fpath.split('/')[2].partition('_')[0]
    hh['time'] = hh['time'].apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return hh

def subtimes(t1,t2):
    return dt.combine(date.min, t1) - dt.combine(dt.min, t2)

def timefromhrs(x):
    minutes = x % 1 * 60.0
    seconds = minutes % 1 * 60
    return time(int(x // 1), int(minutes // 1), int(seconds // 1))

def incrementdate(d, delta=1): #d is a string
    newdate = date(2016, int(d[:2]), int(d[2:])) + timedelta(days=delta)
    return str.zfill(str(newdate.month), 2) + str.zfill(str(newdate.day), 2)

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

def aggalldata():
    walkfiles = sorted(walk('Datasets/'))
    for i, (dirpath, dirnames, filenames) in enumerate(walkfiles):
        if dirnames != []:
            for dirn in sorted(dirnames):
                print dirn.split('/')[-1]
                aggdata(dirn.split('/')[-1])
            break

def aggdata(cdate):

#   read all the data from the specified date, and the date before and after it
    hh = pd.DataFrame()
    folders = [f for f in sorted(os.listdir('Datasets/')) if '.csv' not in f]
    order = folders.index(cdate)
    firstday, lastday = False, False
    for i in [j for j in [order-1,order,order+1] if j >= 0]:
        if i >= len(folders):
            break
        ddate = folders[i]
        # print ddate
        if not os.path.exists('Datasets/' + ddate):
            if i == order - 1:
                firstday = True
            else:
                lastday = True
            print 'No such directory:', 'Datasets/' + ddate
            # print os.listdir('Datasets/' + ddate)
            continue
        
        for fi in [f for f in os.listdir('Datasets/' + ddate) if f[-3:] == 'csv' and f != 'all.csv']:
            fpath = 'Datasets/{}/{}'.format(ddate, fi)
            fd = readdata(fpath)
    #       add random dummy trades
            if 'trade' in fi:
                try:
                    fd = dummytrades(fd)
                except:
                    print fd
                    raise TypeError
            hh = pd.concat([hh, fd], ignore_index=True)

#   days until next 
#   1 except for last day of week, 3 for fridays, 4 for long weekends etc
    if not lastday:
        nextdate = hh['time'].iloc[-1].date()
        hh['daysnext'] = hh['time'].apply(lambda x: (nextdate - x.date()).total_seconds()/86400)
    else:
        hh['daysnext'] = None

#   create a flag for trade or order
    hh.loc[np.isnan(hh['type']),['type']] = 0
    hh['type'] = hh['type'].astype(np.bool)

#   integrate trades and orders, fillna
    hh = hh.sort_values(['ticker','time','type'])
    hh[['bid_price', 'bid_qty', 'offer_price', 'offer_qty']] = hh[['bid_price', 'bid_qty', 'offer_price', 'offer_qty']].fillna(method='ffill')
    hh['trade_price'] = hh['trade_price'].fillna((hh['bid_price'] + hh['offer_price'])/2)
    hh['trade_qty'] = hh['trade_qty'].fillna(0)
    
    # time since the last observation
    hh = hh.sort_values('time')
    hh['time_since_last'] = (hh['time'] - hh['time'].shift()).apply(lambda x: x.total_seconds()) / 234
    # fill in the first "time since last"
    if firstday:
        hh.iloc[0,10] = hh.iloc[0,4] - 102 + 86400 / 234

    # add predictions
    hh = addtargets(hh)

    # just keep the dates that are of concern
    hh = hh[hh['time'].apply(lambda x: x.date()) == date(2016, int(cdate[:2]), int(cdate[2:]))]

    # normalize times, omit the date
    try:
        firsttrade = hh[(hh['type']) & (hh['trade_qty']!=0)]['time'].min().time()
    except:
        print 'ERROR'
        hh.to_csv('Datasets/aaaa.csv')
        raise TypeError
    # hh['date'] = hh['time'].apply(lambda x: (x.date() - date(2016,10,4)).total_seconds() / 86400)
    hh['time'] = hh['time'].apply(lambda x: subtimes(x.time(), firsttrade).total_seconds() / 234)

#   export
    hh.to_csv('Datasets/' + cdate + '/all.csv', index=False)

def addtargets(df, target='JPM', ticksize=0.01, lookahead=0, type1='trade', type2='price'):
    # type1 = trade, bid, offer
    # type2 = price, qty, time
    # lookahead is the amount of time in seconds to wait before checking for the next trade
    dft = (df['ticker'] == target)
    pred = 'next_{}_{}'.format(type1, type2)

    # midpoint
    df.loc[dft,'midpoint'] = (df.loc[dft,'bid_price'] + df.loc[dft,'offer_price']) / 2
    df.loc[:,'midpoint'] = df['midpoint'].ffill()

#---THE PREDICTION
    # real trades are where the prediction trade_qty is not zero
    if type2 == 'trade':
        nextpredcond = (df[dft]['trade_qty'].shift(-1) != 0)
    # real order book updates are different from their previous ones
    else:
        nextpredcond = ((df[dft]['bid_qty'] != df[dft]['bid_qty'].shift(-1)) & (df[dft]['offer_qty'] != df[dft]['offer_qty'].shift(-1)))
    nextpred = df[dft][type1 + '_' + type2].shift(-1)
    
#   fill it in
    df.loc[dft,pred] = np.where(nextpredcond, nextpred, np.nan)
    df.loc[:,pred] = df[pred].bfill()
    df['target'] = (df[pred] - df['midpoint']).apply(lambda x: round2(x, ticksize / 2))
    df.drop([pred,'midpoint'], axis=1, inplace=True)
    return df

def tickerdummies(cdate, tickers = [u'BAC', u'BBD', u'BBT', u'BK', u'BNS', u'BOH', u'BXS', u'C', u'CBU',
       u'CFR', u'CMA', u'COF', u'CPF', u'DB', u'FBP', u'FCF', u'FSB', u'HTH',
       u'ITUB', u'JPM', u'KEY', u'MFG', u'MTB', u'OFG', u'PB', u'PNC', u'RF',
       u'SCNB', u'SHG', u'SNV', u'STI', u'STL', u'STT', u'TCB', u'UBS', u'USB',
       u'VLY', u'WFC'],
       outside=0):
    df = pd.read_csv('Datasets/{}/all.csv'.format(cdate))
    df = pd.concat([df, pd.get_dummies(df['ticker']).astype(np.bool)], axis=1)
    for ticker in tickers:
        if ticker not in df.columns:
            # print 'KKKK', df.columns
            df[ticker] = False
            # print df.iloc[1,:]
    # print df.columns, sorted(df.columns)
    df = df.reindex_axis(sorted(df.columns), axis=1)
    # remove everything outside of 4 minutes of market open and close
    df = df[(df['time'] < 100 * (1 + outside)) & (df['time'] > (-outside))]
    df['time'].iloc[0] = 267.23
    # df.set_value(0,'time',267.23)
    return df

def generateinput(target='JPM', 
                 initialdate='1004', 
                 pricelevels=51,
                 ticksize=0.01,
                 window=20, # how far to unroll, how far we BPTT
                 numrows=256*4, # how many timesteps from the original dataframe we need
                 step=4, # how many timesteps each successive input is
                        # for example if window=3 and step=2 then [1,2,3,4,5] gets unrolled into [1,2,3], [3,4,5]
                 outside=0, # whether to include observations outside of market hours
                 crow=0):
    
    cdate = initialdate
    cfile = tickerdummies(cdate, outside=outside)
    # print cfile.shape
    
#   build the dataset
    while True:
        l = len(cfile)
#       we first get 10% more than we need, and throw away the extras later
        rowuntil = crow + int(numrows * 1.1)
        df = cfile.iloc[crow:rowuntil, :]
#       if the current dataset doesn't contain enough data
        if rowuntil > l:
            rowsleft = rowuntil - l
            cdate = incrementdate(cdate, outside=outside)
            # open the next file
            try:
                cfile = tickerdummies(cdate, outside=outside)
            except:
#               restart the loop
                cdate, crow = initialdate, 0
                cfile = tickerdummies(cdate)
                continue
            df = pd.concat([df, cfile.iloc[:rowsleft]])

#       ticker as a one-hot encoding
        # df = pd.concat([df, pd.get_dummies(df['ticker']).astype(np.bool)], axis=1)
        # print df.columns
        # remove empty predictions and trim the dataset
        df = df[~np.isnan(df['target'])].iloc[:numrows + window]
        # print df.shape
        crow = df.index.max()

        # convert to numpy array
        targets = np.array(df['target'])
        # print df.shape, df.iloc[0,:]
        df = np.array(df.drop(['ticker','target','midpoint','next_trade_price'], axis=1))

#       THE DATA: note that we use zeros to set aside the memory at the outset
        X = np.zeros((numrows / step, window, len(df[0])))
        for i in range(0, numrows / step):
            X[i] = df[i*step:i*step + window, :]
#       puts the targets into bins
        y = makeprob(targets, numrows / step, pricelevels, ticksize, step, window)
        # print X.shape[0], crow
        yield X, y

def round2(number,roundto):
    return (round(number / roundto) * roundto)

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

from bisect import bisect_left

def takeClosest(myList, myNumber):
#     Assumes myList is sorted. Returns closest value to myNumber.
#     If two numbers are equally close, return the smallest number.
#     Credit goes to http://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
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
        
def makeprob(targets, inputlen, pricelevels, ticksize, step, window, tsteps=1):
    
    bins = sorted(list(genbins(pricelevels, ticksize)))
    bindict = {x:i for i, x in enumerate(bins)}
    htick = ticksize / 2
    
    if tsteps > 1:
        raise NotImplementedError

    y = np.zeros((inputlen, pricelevels))
    for i in range(0, inputlen):
        y[i, bindict[takeClosest(bins, targets[i*step + window])]] = 1
    return y

# if __name__ == '__main__':
# 	getinddata(window=1)