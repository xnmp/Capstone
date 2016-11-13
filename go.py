import numpy as np, pandas as pd
np.random.seed = 7
import processing as p
import getdata as g
# import model as m

# from keras.models import Sequential, load_model
# from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


# to do
# ===============

# p.aggalldata(target='JPM')

# p.aggdata('1004', target='JPM')

# ff = pd.read_csv('hhh')

# print res[res>1]

# gen = p.generateinput(initialdate='1004',crow=32700)
# for i in range(3):
# 	X, y = gen.next()
# 	print X.shape

import keras
print keras.__version__