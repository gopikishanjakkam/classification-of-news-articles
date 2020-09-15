#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:45:13 2019

@author: ansh
"""


import numpy as np
from keras import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



def get_tfidf_features(df, test_split=0.2, one_hot=True):

    X_train, X_test, y_train, y_test = train_test_split(df.situation, df.label_id,
                                                    test_size=test_split, shuffle=True)

    print ("Train size: %d" % X_train.shape[0])
    print ("Test size: %d" % X_test.shape[0])
    
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                            ngram_range=(1, 2), stop_words='english')


    print("Generating tfidf feature vectors... fitting on train docs..")
    train_features = tfidf.fit_transform(X_train).toarray()
    print("Transforming test docs using tfidf...")
    test_features = tfidf.transform(X_test)
    print("Generated feature vectors of size ", train_features.shape)


    encoder = LabelEncoder()
    encoder.fit(y_train.values)
    train_labels = encoder.transform(y_train)
    test_labels = encoder.transform(y_test)

    if(one_hot):
        num_classes = np.max(train_labels) + 1
        train_labels = utils.to_categorical(train_labels, num_classes)
        test_labels = utils.to_categorical(test_labels, num_classes)


    print('Train features shape:\t', train_features.shape)
    print('Train labels shape:\t', train_labels.shape)
    print('Test features shape:\t', test_features.shape)
    print('Test labels shape:\t', test_labels.shape)
    
    return train_features, test_features, train_labels, test_labels


'''

usage:
    
x_train, x_test, y_train, y_test = TFIDF.get_tfidf_features(pd.read_csv( \
                                    '../dataset/cleaned_dataset_large_all.csv'))


'''