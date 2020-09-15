#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 02:20:21 2019

@author: ansh
"""
import re
import string
import requests
import pickle
import xgboost as xgb
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from bs4 import BeautifulSoup
from bs4.element import Comment


import spacy
import stanfordnlp

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)



# Cleaning the text sentences so that punctuation marks, stop words &amp; digits are removed
def clean_text(doc, lower=False, url_token='URL', fin_fig_token='MONEY', keep_date=True):
    
    if lower: doc = doc.lower()
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
#    date_removed = ' '.join([word for word in punc_free.split() if word.lower() not in months])
    date_removed = re.sub(r"\b(19|20)\d{2}\b", 'YEAR', punc_free)
    # lemmatize
#    normalized = ' '.join(lemma.lemmatize(word) for word in date_removed.split())
#    stemmed = ' '.join([stemmer(word) for word in date_removed.split()])
    
    # remove all remaining digits
    digits_removed = re.sub(r"\d+", ' ', date_removed)
    out = ' '.join([x for x in digits_removed.split()])

    return out




def get_doc_from_url(url):
    response = requests.get(url)
    if (response.status_code == 200):
        soup = BeautifulSoup(response.text, "html.parser")  
        doc_text = soup.find_all(['p', 'tr']) # fixed code to extract text from tables as well.
        doc_text = ''.join([x.text for x in doc_text])
    cleaned_doc = clean_text(doc_text, lower=True)
    return cleaned_doc


def live_classify(url):
    
    tokenizer_handle = 'model/tokenizer.pickle'
    encoder_handle = 'model/encoder.pickle'
    model_handle = 'model/xgb_tr0.96_te0.87_tr0.95.model'

    doc = [get_doc_from_url(url)]
    with open(tokenizer_handle, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(encoder_handle, 'rb') as handle:
        encoder = pickle.load(handle)

    x=tokenizer.texts_to_matrix(doc, mode='tfidf')
    d = xgb.DMatrix(data=x)

    bst = xgb.Booster({'nthread': 1})  # init model
    bst.load_model(model_handle)
    return (encoder.inverse_transform([int(bst.predict(d)[0])])[0])








def get_org_name(url):

    response = requests.get(url)
    if (response.status_code == 200):
        soup = BeautifulSoup(response.text, "html.parser")  
        doc_text = soup.find_all(['p', 'tr']) # fixed code to extract text from tables as well.
        doc_text = ''.join([x.text for x in doc_text])
    doc = clean_text(doc_text, lower=False)
    
    nlp = spacy.load('en_core_web_sm') 
    stf_nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')
    doc = doc[:250].split()
    doc =  ' '.join([x for x in doc if x.lower().find('newswire') == -1])

    orgs = set()
    spacy_doc = ' '.join(doc.split()).strip()
    nlp_doc = nlp(spacy_doc)
    for ent in nlp_doc.ents: 
        if(ent.label_ == 'ORG'):
            orgs.add(ent.text) 
    
    sf_doc = stf_nlp(doc)

    for sent in sf_doc.sentences:
        for word in sent.words:
            if(word.xpos=='NNP' and word.upos=='PROPN'):
                orgs.add(word.text)
    
    return ', '.join(list(orgs))