#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:57:48 2019

@author: ansh
"""

import sys
import utils
import pickle
import xgboost as xgb

url=sys.argv[1]

tokenizer_handle = 'model/tokenizer.pickle'
encoder_handle = 'model/encoder.pickle'
model_handle = 'model/xgb_tr0.96_te0.87_tr0.95.model'

doc = [utils.get_doc_from_url(url)]
with open(tokenizer_handle, 'rb') as handle:
    tokenizer = pickle.load(handle)
with open(encoder_handle, 'rb') as handle:
    encoder = pickle.load(handle)


x=tokenizer.texts_to_matrix(doc, mode='tfidf')
d = xgb.DMatrix(data=x)


bst = xgb.Booster({'nthread': 1})  # init model
bst.load_model(model_handle)
ypred = int(bst.predict(d)[0])
print(encoder.inverse_transform([ypred])[0])
