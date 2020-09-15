#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:53:20 2019

@author: ansh
"""

import pandas as pd
import utils
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

df = pd.read_csv('live_classification_url.csv')

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')










###############################################################################





classes = ['contracts', 'dividend', 'earnings', 'joint_venture',
       'labor_union', 'lawsuit', 'mgmt_changes', 'product']


f = open('live_classify_results_with_orgs.txt', 'w')

true = []
pred = []
orgs = []
print('True Labels\tPredicted Lables\n###########\t################', file=f, flush=True)
for row in df.iterrows():
    ix=row[0]
    row=row[1]
    true_label = row[0]
    url = row[1]
    predicted_label = utils.live_classify(url)
    org = utils.get_org_name(url)
    
    orgs.append(org)
    true.append(true_label)
    pred.append(predicted_label)
    print(str(ix)+' '+true_label+'\t'+predicted_label+'\t'+org, file=f, flush=True)
    time.sleep(0.5)
    
print('classification ACCURACY SCORE' , accuracy_score(true, pred), file=f, flush=True)




cnf_matrix = confusion_matrix(true, pred)
plot_confusion_matrix(cnf_matrix, classes=classes, title="Confusion matrix")

from sklearn.metrics import classification_report
rep = classification_report(true, pred)
print(rep)
