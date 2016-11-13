# retrieve data
import processing as p, model as m
import numpy as np, pandas as pd
from itertools import product


from keras.models import Sequential
from keras.models import load_model


# model = load_model('model.h5')

gen = p.generateinput(initialdate='1104')

model.predict_generator(gen, val_samples=1000)



scores = model.evaluate(X, y)