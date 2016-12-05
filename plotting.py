# from runmodel import modelpredict
# from benchmark import benchmarkpreds
import processing as p
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import pearsonr
from aux import *
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mabs

basecols = ['time','bid_price','bid_qty','offer_price','offer_qty',
            'trade_price','trade_qty','midpoint','spread']
graphcols = ['midpoint_change','trade_price_d','next_trade_price','target']
predcols = ['type','timeopen','time_since_last','oddeven']

def plotr(ax, lbound=40.9, ubound=41.2, var2='time', dist=True):
    subset = preds[(lbound < preds['time']) & (ubound > preds['time'])]
    realtrades = subset[subset['trade_qty'] > 0]
    
    # plt.hist(subset['target'], bins=[i*0.005-0.0275 for i in range(13)])
    # plt.show()
    
    print 'pearsonr: ', pearsonr(subset['inv_weighted_d_pred'], subset['target'])
    print 'pearsonr: ', pearsonr(subset['model_d_pred'], subset['target'])
    print 'pearsonr: ', pearsonr(subset['Gradboost_d_pred'], subset['target'])
    print 'mse inv weighted midpoint: ', mabs(subset['inv_weighted_d_pred'], subset['target'])
    print 'mse model: ', mabs(subset['model_d_pred'], subset['target'])
    print 'mse benchmark: ', mabs(subset['Gradboost_d_pred'], subset['target'])
    
    # print 'ENTROPY', realtrades['entropy'].mean()
    
    if dist:
        # the probability distribution
        probs = subset.loc[:, ['time','midpoint'] + map(str,range(30))]
        probs = pd.melt(probs, id_vars=['time','midpoint'], 
                        value_vars=map(str,range(6,25)), value_name='prob')
        
        bins = sorted(list(genbins(31, 0.01)))
        bindict = {str(i):x for i, x in enumerate(bins)}
        probs['variable'] = probs['variable'].map(bindict.get)
        probs = probs.rename(columns={'variable':'price_d'})
        
        ax.scatter(probs['time'],probs['midpoint']+probs['price_d'],
                   s=80,c=np.sqrt(probs['prob']),marker='s',cmap='OrRd',alpha=0.8,
                   lw=0.0)
    
    # bids, offers and trades
    ax.scatter(subset['time'], subset['trade_price'], s=100*np.sqrt(subset['trade_qty']), 
               color='k', alpha=0.5)
    ax.plot(subset['time'], subset['bid_price'],
            color='k', linewidth=1)
    ax.plot(subset['time'], subset['offer_price'],
            color='k', linewidth=1)
    
    if not dist:
        ax.fill_between(subset['time'], subset['plot_bid_qty'], subset['bid_price'], 
                        color=(0.7,0.7,0.7), alpha=0.5)
        ax.fill_between(subset['time'], subset['plot_offer_qty'], subset['offer_price'], 
                        color=(0.7,0.7,0.7), alpha=0.5)
        
        # the comparators
        iw, = ax.plot(subset['time'], subset['inv_weighted_pred'],
                       color='g', linewidth=3)
        bm, = ax.plot(subset['time'], subset['Gradboost_pred'],
                       color='m', linewidth=2)
        m, = ax.plot(subset['time'], subset['model_pred'],
                      color='b', linewidth=2)
        ntp, = ax.plot(subset['time'], subset['next_trade_price'], 
                       color='r', linewidth=2)
        # legend
        # ax.legend([ntp, iw, bm, m], ['Target','Inverse Weighted','Benchmark','Model'])
    
    # # entropy
    # level = subset['offer_price'].max() + 0.01
    # ent, = ax.plot(subset['time'], subset['entropy'] * 0.005 + level, 
    #                color='k', linewidth=2, linestyle='dotted')
    # ax.scatter(subset['time'], subset['trade_price'] * 0 + level, s=100*np.sqrt(subset['trade_qty']), 
    #            color='k')
    # ax.axhline(y=level, xmin=0, xmax=500, color='k', linewidth=0.5)
    # ax.axhline(y=level+0.005, xmin=0, xmax=500, color='k', linewidth=0.5)
    
def plotd(ax, lbound=40.9, ubound=41.2, var2='time', dist=True):
    subset = preds[(lbound < preds['time']) & (ubound > preds['time'])]
    realtrades = subset[subset['trade_qty'] > 0]
    
    if dist:
      # the probability distribution
      probs = subset.loc[:, ['time','midpoint'] + map(str,range(30))]
      probs = pd.melt(probs, id_vars=['time','midpoint'], 
                      value_vars=map(str,range(6,25)), value_name='prob')
      
      bins = sorted(list(genbins(31, 0.01)))
      bindict = {str(i):x for i, x in enumerate(bins)}
      probs['variable'] = probs['variable'].map(bindict.get)
      probs = probs.rename(columns={'variable':'price_d'})
      
      ax.scatter(probs['time'], probs['price_d'],
                 s=80,c=np.sqrt(probs['prob']),marker='s',cmap='OrRd',alpha=0.8,
                 lw=0.0)
      
    # bids, offers and trades
    ax.scatter(subset['time'], subset['trade_price_d'], s=100*subset['trade_qty'], 
               color='k')
    ax.plot(subset['time'], subset['midpoint'] - subset['midpoint'].min() + 0.05, 
            color='r', ls='dashed')
    ax.plot(subset['time'], subset['spread'] / 2,
            color='k', linewidth=1)
    ax.plot(subset['time'], -subset['spread'] / 2,
            color='k', linewidth=1)
    # ax.fill_between(subset['time'], subset['plot_bid_qty_d'], -subset['spread'] / 2, 
    #                 color=(0.7,0.7,0.7), alpha=0.5)
    # ax.fill_between(subset['time'], subset['plot_offer_qty_d'], subset['spread'] / 2, 
    #                 color=(0.7,0.7,0.7), alpha=0.5)
    
    # entropy
    level = subset['spread'].max() / 2 + 0.05
    ent, = ax.plot(subset['time'], subset['entropy'] * 0.005 + level, 
                   color='k', linewidth=2, linestyle='dotted')
    ax.scatter(subset['time'], subset['trade_price'] * 0 + level, s=100*subset['trade_qty'], 
               color='k')
    ax.axhline(y=level, xmin=0, xmax=500, color='k', linewidth=0.5)
    ax.axhline(y=level+0.005, xmin=0, xmax=500, color='k', linewidth=0.5)
    
    # # the comparators
    # iw, = ax.plot(subset['time'], subset['inv_weighted_d_pred'],
    #                color='g', linewidth=2)
    # bm, = ax.plot(subset['time'], subset['Gradboost_d_pred'],
    #                color='m', linewidth=2)
    # m, = ax.plot(subset['time'], subset['model_d_pred'],
    #               color='b', linewidth=2)
    # ntp, = ax.plot(subset['time'], subset['target'],
    #                color='r', linewidth=2)
    
    # ax.legend([ntp, iw, bm, m], ['Target','Inverse Weighted','Benchmark','Model'])

def compete(df, method1, method2):
    df['traded'] = (df[method1 + '_pred'] + df[method2 + '_pred']) / 2
    pricediff = df['next_trade_price'] - df['traded'] 
    df['pnl1'] = np.where(df[method1 + '_pred'] > df['traded'], 
                          pricediff,
                          -pricediff)
    df['pnl2'] = np.where(df[method2 + '_pred'] > df['traded'], 
                          pricediff,
                          -pricediff)
    df['cum_pnl1'] = df['pnl1'].cumsum()
    df['cum_pnl2'] = df['pnl2'].cumsum()
    return df[basecols + graphcols + [method1 + '_pred', method2 + '_pred', 
              'pnl1', 'pnl2', 'cum_pnl1', 'cum_pnl2']]

def plotbatchloss():
    # generates plots for the intra-epoch loss
    fig, ax1 = plt.subplots(1,1)

    l = {}
    totl = 0
    for d in datelist2('1004', '1111'):
        l[d] = epochlength(idate=d, ldate=d) / 64
        totl += l[d]

    for i in range(1, 35):
        lossh = pd.read_csv('Results/loss_history{}.csv'.format(i), index_col=0, header=None, names=['loss'])
        numobs = len(lossh)
        fig = plt.gcf()
        fig.set_size_inches(8,4.5)
        plt.ylim(0,5)
        plt.plot(lossh)
        
        suml = 0
        for d in datelist2('1004', '1111'):
            # l = epochlength(idate=d, ldate=d) / 64
            suml += l[d]
            if l[d] > 0 and suml <= numobs:
                labelloc = (suml-(i-1)*235)
                if labelloc < 0:
                    labelloc += totl
                plt.axvline(x=labelloc, ymin=0, ymax=1, color='k', linewidth=1.5, linestyle='dotted')
                plt.xlabel('Batch Number')
                plt.ylabel('Loss')
                plt.annotate(d, xy=(labelloc, 0.4), xytext=(labelloc-20, 0.4), rotation=90, color='r')
        plt.savefig('Presentations/Graphs/lossh_{}.png'.format(i))
        plt.gcf().clear()

# mpreds = pd.read_csv('Results/predsall.csv', index_col=0)

# # modify the index
datedict = {x:i for i,x in enumerate(datelist2('1004','1118'))}
# oldcol = mpreds.columns
# mpreds = mpreds.reset_index()
# mpreds.columns = ['obid'] + list(oldcol)
# mpreds['date'] = mpreds['date'].apply(lambda x: dtm.datetime.strptime(x, '%Y-%m-%d'))
# mpreds['obid'] = mpreds['obid'] + mpreds['date'].apply(lambda x: datedict[dtm.date.strftime(x, '%m%d')])*100000
# mpreds.set_index('obid', inplace=True)
# mpreds['time'] = -3000 + mpreds['time'] + mpreds['date'].apply(lambda x: datedict[dtm.date.strftime(x, '%m%d')])*100


# bpreds = pd.read_csv('Results/benchmark_predsall.csv', index_col=0)#[:31168]
# bpreds = bpreds.iloc[:bpreds.shape[0] / 64 * 64, :]
# preds = pd.merge(mpreds,
#                  bpreds.drop(basecols + graphcols, axis=1), 
#                  how='inner', left_index=True, right_index=True)


# preds['inv_weighted_d_pred'] = preds['inv_weighted_pred'] - preds['midpoint']
# preds['Gradboost_d_pred'] = preds['Gradboost_pred'] - preds['midpoint']
# # preds['plot_bid_qty'] = preds['bid_price'] - preds['bid_qty'] * 0.01
# # preds['plot_offer_qty'] = preds['offer_price'] + preds['offer_qty'] * 0.01
# # preds['plot_bid_qty_d'] = preds['spread'] / 2 - preds['bid_qty'] * 0.01
# # preds['plot_offer_qty_d'] = preds['spread'] / 2 + preds['offer_qty'] * 0.01

# # maxentrop = mpreds['time'].loc[mpreds['roll_entropy'].idxmax()]
# # maxstd = mpreds['time'].loc[mpreds['roll_std'].idxmax()]
# # maxmomentum = mpreds['time'].loc[mpreds['roll_momentum_actual'].idxmax()]
# # minmomentum = mpreds['time'].loc[mpreds['roll_momentum_actual'].idxmin()]

# fig, ax1 = plt.subplots(1,1)
# # plotr(ax1, lbound=45, ubound=47, dist=False)


preds = pd.read_csv('aaa.csv')

print 'pearsonr inv: ', pearsonr(preds['inv_weighted_d_pred'], preds['target'])
print 'pearsonr model: ', pearsonr(preds['model_d_pred'], preds['target'])
print 'pearsonr gradboost: ', pearsonr(preds['Gradboost_d_pred'], preds['target'])
print 'mse inv weighted midpoint: ', mabs(preds['inv_weighted_d_pred'], preds['target'])
print 'mse model: ', mabs(preds['model_d_pred'], preds['target'])
print 'mse benchmark: ', mabs(preds['Gradboost_d_pred'], preds['target'])


# # pnl plots
# go1 = compete(preds, 'model', 'Gradboost')
# go2 = compete(preds, 'model', 'inv_weighted')

# bm, = plt.plot(go1['time'], go1['cum_pnl1'])
# inv, = plt.plot(go2['time'], go2['cum_pnl1'])

# plt.xlabel('time')
# plt.ylabel('dollars')

# plt.legend([inv, bm], ['Inverse Weighted','Benchmark'])

# for i in range(4):
#     plt.axvline(x=i*100, ymin=0, ymax=100, color='k', linewidth=1.5, linestyle='dotted')

# plt.show()


# # error variables
# rwind = 30
# preds['ii'] = range(len(preds))
# preds['roll_pearsonr_bm'] = preds.ii.rolling(rwind).apply(lambda x: rollpearson(x, preds, var='Gradboost_d_pred')).fillna(0)
# # preds['roll_mse_bm'] = preds.ii.rolling(rwind).apply(lambda x: rollmse(x, preds, var='Gradboost_d_pred')).fillna(-1)
# preds['roll_pearsonr_inv'] = preds.ii.rolling(rwind).apply(lambda x: rollpearson(x, preds, var='inv_weighted_d_pred')).fillna(0)
# preds = preds.drop(['ii'], axis=1)
# preds.to_csv('aaa.csv')

preds = pd.read_csv('aaa.csv')

lbound, ubound = 40,50
preds = preds[(lbound < preds['time']) & (ubound > preds['time'])]

rwind = 30

preds['se_m'] = (preds['target'] - preds['model_d_pred'])**2
preds['se_bm'] = (preds['target'] - preds['Gradboost_d_pred'])**2
preds['se_inv'] = (preds['target'] - preds['inv_weighted_d_pred'])**2
preds['roll_se_m'] = np.sqrt(preds['se_m'].rolling(rwind).mean())
preds['roll_se_bm'] = np.sqrt(preds['se_bm'].rolling(rwind).mean())
preds['roll_se_inv'] = np.sqrt(preds['se_inv'].rolling(rwind).mean())

rwind2 = 30

preds['roll_pr_inv'] = preds['roll_pearsonr_inv'].rolling(rwind2).mean()
preds['roll_pr_bm'] = preds['roll_pearsonr_bm'].rolling(rwind2).mean()
preds['roll_pr_m'] = preds['roll_pearsonr'].rolling(rwind2).mean()

preds['roll_se2_inv'] = preds['roll_se_inv'].rolling(rwind2).mean()
preds['roll_se2_bm'] = preds['roll_se_bm'].rolling(rwind2).mean()
preds['roll_se2_m'] = preds['roll_se_m'].rolling(rwind2).mean()


# lbound, ubound = 40, 50
# preds = preds[(lbound < preds['time']) & (ubound > preds['time'])]

f, axarr = plt.subplots(2, sharex=True)

# error plots - pearson r
inv, = axarr[0].plot(preds['time'], preds['roll_pr_inv'])
bm, = axarr[0].plot(preds['time'], preds['roll_pr_bm'])
m, = axarr[0].plot(preds['time'], preds['roll_pr_m'])
axarr[0].legend([inv, bm, m], ['Inverse Weighted', 'Benchmark', 'Model'])
axarr[0].set_ylabel('Pearson R')

# for i in range(4):
#     axarr[0].axvline(x=i*100, ymin=0, ymax=100, color='k', linewidth=1.5, linestyle='dotted')
#     axarr[0].annotate(datelist2('1114','1118')[i], xy=(i*100, 0.4), xytext=(i*100, 0.4), rotation=90, color='k')
    


# mse
inv, = axarr[1].plot(preds['time'], preds['roll_se2_inv'])
bm, = axarr[1].plot(preds['time'], preds['roll_se2_bm'])
m, = axarr[1].plot(preds['time'], preds['roll_se2_m'])
axarr[1].legend([inv, bm, m], ['Inverse Weighted', 'Benchmark', 'Model'])
axarr[1].set_ylim((0,0.02))
axarr[1].set_ylabel('Rolling RMSE')
axarr[1].set_xlabel('Time')



# for i in range(4):
#     axarr[1].axvline(x=i*100, ymin=0, ymax=100, color='k', linewidth=1.5, linestyle='dotted')
#     axarr[1].annotate(datelist2('1114','1118')[i], xy=(i*100, 0.4), xytext=(i*100, 0.4), rotation=90, color='k')
    
plt.show()