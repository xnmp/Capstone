import io, requests, os
import numpy as np, pandas as pd
from datetime import datetime as dt, timedelta, time, date
from itertools import product
from os import walk
from aux import *
from bisect import bisect_left

# -----------
# RETREIVE DATA
# -----------

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


# if __name__ == '__main__':
#   getinddata(window=1)