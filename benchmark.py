import numpy as np, pandas as pd
from processing import generateinput, preparedata
from aux import *

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mabs
from sklearn.metrics import r2_score as r2
from sklearn.preprocessing import RobustScaler
from sklearn.externals import joblib
from scipy.stats import pearsonr

basecols = ['time','bid_price','bid_qty','offer_price','offer_qty',
         'trade_price','trade_qty','midpoint','spread']
graphcols = ['midpoint_change','trade_price_d','next_trade_price','target']
predcols = ['type','timeopen','time_since_last','oddeven']

# THE MODELS

linSVR_params = {'kernel':['linear'],
                 'C':[1+x/2 for x in range(4)]}
rbfSVR_params = {'kernel':['rbf'],
                 'C':[1+x/2 for x in range(4)],
                 } # 'gamma':[x/20 for x in range(1, 5)]
forest_params = {'n_estimators': range(7,20),
                 'max_depth': range(2,10)}
ada_params = {'n_estimators': range(20,70,10)}
gradboost_params = {'n_estimators': range(70,130,20)}
bag_params = {'n_estimators': range(8,20,2),
              'max_features': [0.4,0.6,0.8,1]}

Ests = {}
# Ests['SVM_Linear'] = GridSearchCV(SVR(), linSVR_params)
# Ests['SVM_RBF'] = GridSearchCV(SVR(), rbfSVR_params)
Ests['Gradboost'] = GridSearchCV(GradientBoostingRegressor(min_samples_split=5), gradboost_params)
# Ests['Bagged'] = GridSearchCV(BaggingRegressor(), bag_params)
# Ests['Adaboost'] = GridSearchCV(AdaBoostRegressor(), ada_params)
# Ests['Random_Forest'] = GridSearchCV(RandomForestRegressor(min_samples_split=10), forest_params)

def benchmarkprepdata(idate='1004', ldate='1111', outside=0):
    
    cols = ['time','bid_price','bid_qty','offer_price','offer_qty','date',
             'trade_price','trade_qty','midpoint','spread',
             'midpoint_change','trade_price_d','next_trade_price','target',
             'type','timeopen','time_since_last','oddeven']
    
    print "Importing Data..."
    # import all the data
    df = pd.DataFrame()
    for date in datelist2(idate, ldate):
        try:
            hh = preparedata(idate, outside=0.2)[cols]
        except:
            continue
        df = pd.concat([df, hh])
    
    # make all of the lagged values
    window = 5
    for col in df.columns.drop(['target','next_trade_price','oddeven','date']):
        for i in range(1, window + 1):
            df[col + str(i)] = df[col].shift(i)
    
    # trim dataset
    df = df.dropna()
    df = df[(df['time'] <= 100 * (1 + outside)) & (df['time'] >= (-outside))]
    X = df.drop(['oddeven','next_trade_price','target','date'], axis=1)
    y = df['target']
    df = df[basecols + graphcols + ['date']]
    
    # df.to_csv('aaa.csv')
    
    # print np.isnan(df.drop(['date'],axis=1))
    
    # apply a robust scaler
    X = RobustScaler().fit_transform(X)

    return df, X, y

def benchmarkfit(idate='1004', ldate='1111', outside=0):

    df, X, y = benchmarkprepdata(idate=idate, ldate=ldate, outside=outside)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # do the fitting
    preds, mses, mabss, r2s, pearsonrs = {}, {}, {}, {}, {}
    for Est_name, Est in Ests.items():
        print 'Fitting', Est_name, '...'
        Est.fit(X_train, y_train)
        joblib.dump(Est.best_estimator_, 'Results/{}.pkl'.format(Est_name))
        # metrics
        # preds[Est_name] = Est.predict(X_test)
        # mses[Est_name] = mse(preds[Est_name], y_test)
        # mabss[Est_name] = mabs(preds[Est_name], y_test)
        # r2s[Est_name] = r2(preds[Est_name], y_test) #This has a problem
        # pearsonrs[Est_name] = pearsonr(preds[Est_name], y_test)[0]
    
    # scores = pd.DataFrame([mabss, mses, r2s, pearsonrs]).transpose()
    # scores.columns = ['MAbs Error', 'MSE', 'R2', 'Pearson R']
    # scores.to_csv('benchmark_scores.csv')
    # print "Exported scores to 'benchmark_scores.csv'"

def benchmarkpredict(idate='1004', ldate='1118', outfile='Results/benchmark_preds.csv'):
    
    datedict = {x:i for i,x in enumerate(datelist2('1004','1118'))}
    df = pd.DataFrame()
    
    for d in datelist2(idate, ldate):
        hh, X, y = benchmarkprepdata(idate=d, ldate=d)
        
        # add predictions to the dataset
        for Est_name, Est in Ests.items():
            print 'Date:', d, 'Predicting', Est_name, '...'
            hh[Est_name + '_pred'] = Est.predict(X) + hh['midpoint']
        
        # modify the index
        oldcol = hh.columns
        hh = hh.reset_index()
        hh.columns = ['obid'] + list(oldcol)
        hh['date'] = hh['date'].apply(lambda x: dtm.datetime.strptime(x, '%Y-%m-%d'))
        hh['obid'] = hh['obid'] + hh['date'].apply(lambda x: datedict[dtm.date.strftime(x, '%m%d')])*100000
        hh.set_index('obid', inplace=True)
        
        hh.to_csv('Results/benchmark_preds{}.csv'.format(d))
        df = pd.concat([df, hh])
    df.to_csv('Results/benchmark_predsall.csv')
    return df

benchmarkfit(idate='1004', ldate='1007')

for Est_name in ['Gradboost']:
    Ests[Est_name] = joblib.load('Results/{}.pkl'.format(Est_name)) 

benchmarkpredict(idate='1111', ldate='1118')
