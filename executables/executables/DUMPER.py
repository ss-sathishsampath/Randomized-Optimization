# -*- coding: utf-8 -*-
"""
@author : Sathish Sampath(ss.sathishsampath@gmail.com)

"""


import numpy as np

import sklearn.model_selection as ms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
adult = pd.read_hdf('datasets.hdf','adult')        
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values





adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     




pipe = Pipeline([('Scale',StandardScaler())]) 


# pipe = Pipeline([('Scale',StandardScaler()),
#                  ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                  ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                  ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                  ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),])
trgX = pipe.fit_transform(adult_trgX,adult_trgY)
trgY = np.atleast_2d(adult_trgY).T
tstX = pipe.transform(adult_tstX)
tstY = np.atleast_2d(adult_tstY).T
trgX, valX, trgY, valY = ms.train_test_split(trgX, trgY, test_size=0.2, random_state=1,stratify=trgY)     
tst = pd.DataFrame(np.hstack((tstX,tstY)))
trg = pd.DataFrame(np.hstack((trgX,trgY)))
val = pd.DataFrame(np.hstack((valX,valY)))
tst.to_csv('a_test.csv',index=False,header=False)
trg.to_csv('a_trg.csv',index=False,header=False)
val.to_csv('a_val.csv',index=False,header=False)