#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:47:23 2019

@author: ansh
"""

import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split


dataset_path = '../dataset/dataset_large_all.csv'
#dataset_path = '../dataset/dataset_test_10.csv'


df = pd.read_csv(dataset_path)
 
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stem = PorterStemmer() 

our_tags = [
        'MONTH',
        'YEAR',
        'MONEY',
        'URL',
        ]

months = ['january', 'february', 'march', 'april', 'may', 'june',
          'july', 'august', 'september', 'october', 'november', 'december',
          'jan','feb','mar','apr','may','jun','jul','aug','sept','oct','nov','oct','dec']

#%%

#stemmer =  lambda x : x[:len(stem.stem(x))]
# Cleaning the text sentences so that punctuation marks, stop words &amp; digits are removed
def clean_text(doc, lower=False, url_token='URL', fin_fig_token='MONEY', keep_date=True):
    
    if lower: doc = doc[10:].lower()
    else: doc = doc[10:]
    #remove email like tokens
    email_removed = ' '.join([item for item in doc.split() if '@' not in item])
    #remove urls
    url_removed = re.sub(r"http[s]?:\/{2}[^\s]*", 'URL', email_removed)
    url_removed = re.sub(r"\w*www.\w*.\w*", 'URL', url_removed)
    # remove financial figures
    fin_fig_removed = re.sub(r"\$\d+[.,]?\d*", 'MONEY', url_removed)
    
    # remove document source name
    source_removed = fin_fig_removed.replace('Share this article', ' ')
    source_removed = source_removed.replace('PRNewswire', '').replace('globe newswire', '')
    
    tokenized = word_tokenize(source_removed)
    # remove stop-words
    stop_free = " ".join([i for i in tokenized if i.lower() not in stop])
    # remove punctuations
    punc_free = stop_free.replace('-', ' ')
    punc_free = ''.join(ch for ch in punc_free if ch not in exclude)
    # remove months & years
    date_removed = ' '.join([word for word in punc_free.split() if word.lower() not in months])
    date_removed = re.sub(r"\b(19|20)\d{2}\b", 'YEAR', date_removed)
    # lemmatize
#    normalized = ' '.join(lemma.lemmatize(word) for word in date_removed.split())
#    stemmed = ' '.join([stemmer(word) for word in date_removed.split()])
    
    # remove all remaining digits
    digits_removed = re.sub(r"\d+", ' ', date_removed)
    out = ' '.join([x for x in digits_removed.split()])

    return out

def clean_headline(doc):
    lower = doc.lower()[11:]
    digits_removed = re.sub(r"\d+", ' ',lower)
    punc_free = ''.join(ch for ch in digits_removed if ch not in exclude)
    out = ' '.join([x for x in punc_free.lower().split()])
    return out

#%%
   
index_label_map = {1:'dividend',
                   2:'contracts',
                   3:'joint_venture',
                   4:'management_changes',
                   5:'lawsuit',
                   6:'product',
                   7:'labor_union',
                   8:'earnings'
                   }

label_index_map = {v: k for k, v in index_label_map.items()}

 
with open('../dataset/classLabel_Index_mapping.txt', 'w') as f:
    print(label_index_map, file=f)


#%%
cleaned_situation = [clean_text(x, lower=True) for x in df.situation]
cleaned_headline = [clean_headline(x) for x in df.headline]

cleaned_df = pd.DataFrame({'label':df['label'], 'headline':cleaned_headline, 'situation':cleaned_situation})
#cleaned_df['label_id'] = cleaned_df.label.map(label_index_map)   

cleaned_df.to_csv('../dataset/cleaned_dataset_large_all_classification_final.csv', index=False)
#cleaned_df.to_csv('../dataset/cleaned_dataset_test_10.csv', index=False)


#%%

train_df, test_df = train_test_split(cleaned_df, train_size=0.8, test_size=0.2, shuffle=True)

print('train df size:', train_df.shape)
print('test df size:', test_df.shape)