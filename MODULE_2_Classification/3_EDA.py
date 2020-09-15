#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 02:12:47 2019

@author: ansh
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

dataset = '../dataset/cleaned_dataset_large_all.csv'
df = pd.read_csv(dataset)

#%%
'''
plot # docs in each category
'''
df.groupby('label').label_id.count().plot.barh()
plt.xlabel('# documents in each category')
plt.ylabel('category')
plt.title('Count of documents in each category')
plt.savefig('./analysis_data/document_class_count.png', quality=100, dpi=200, bbox_inches='tight')
#%%
'''
histogram of document length
'''
doc_len = [x.__len__() for x in df.situation]
plt.hist(doc_len, bins='auto')
plt.title('histogram of document length')
plt.xlabel('document length')
plt.ylabel('# documents')
plt.show()



#%%
'''
extract features
'''
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.situation).toarray()
labels = df.label_id
features.shape

#%%


category_id_df = df[['label', 'label_id']].drop_duplicates().sort_values('label_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['label_id', 'label']].values)


from sklearn.feature_selection import chi2

N = 10
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#  trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
#  print("  . Most correlated trigrams:\n       . {}".format('\n       . '.join(trigrams[-N:])))
  
  
#%%
'''
TSNE
'''
from sklearn.manifold import TSNE

# Sampling a subset of our dataset because t-SNE is computationally expensive
SAMPLE_SIZE = int(len(features) * 0.3)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])


for category, category_id in sorted(category_to_id.items()):
    points = projected_features[(labels[indices] == category_id).values]
#    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
    plt.scatter(points[:, 0], points[:, 1], s=4, label=category)


plt.title("tf-idf feature vector for each article, projected on 2 dimensions.")
plt.legend(loc=7, bbox_to_anchor=(1.5, 0.3), ncol=1)

plt.savefig('./analysis_data/tsne_plot.png', quality=100, dpi=200, bbox_inches='tight')
  
  
#%%

  
  
  
  
  