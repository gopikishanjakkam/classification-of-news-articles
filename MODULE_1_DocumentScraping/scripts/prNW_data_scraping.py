#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:08:14 2019

@author: ansh
"""

'''
This script will extract the document text from the prnewswire URLs.
The URLs are saved in a csv file for 5 classification topics, totalling to about 2000 URLs.

Adviced to run multiple copies of this script after changing the START_RANGE & END_RANGE values.
eg:
START_RANGE = 0 ; END_RANGE = -1; will scrap all URLs
START_RANGE = 500 ; END_RANGE = 1000; will scrap URL index 500 to 1000 from the csv.

NOTE:   To avoid scraping same document multiple times because of multiple runs of script with overlapping ranges,
        script will not fetch document if a file corresponding to it's filename already exists.

NOTE:   The extrated documents will be saved in data/PR_data/<classification_topic_name>/<doc_ix>.txt
'''


import os
import requests
from os import path
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment


SAVE_PATH = '../../scrapped_data/PR_data/'
START_RANGE = 0
END_RANGE = -1



url_file = '../prNW-scrapped-URLs-500.csv'
url_df = pd.read_csv(url_file)


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)


for row in url_df.iloc[START_RANGE:END_RANGE].iterrows():
    metadata = row[1]
    topic_dir = SAVE_PATH + metadata.topics + '/'
    filename = topic_dir + str(row[0]) + '.txt'
    
    if (path.exists(filename)):
        print(' data from url {} exists, skipping!'.format(metadata.URL))
        continue;
        
    print('FETCHING document from url {} & SAVING in file {}'.format(metadata.URL, filename))
    response = requests.get(metadata.URL)
    if (response.status_code == 200):
        soup = BeautifulSoup(response.text, "html.parser")  
        doc_text = text_from_html(soup.text).replace('\n\n', '')
        doc_text = doc_text[doc_text.find('News provided by')+16 : doc_text.find('Related Links')]
        
        if not os.path.exists(topic_dir): os.system('mkdir -p ' + topic_dir)
        
        with open(filename, 'w') as f:
            f.write('HEADLINE : {}\n'.format(metadata.document_title))
            f.write('SITUATION : ' + doc_text)
    else:
        print('fetching failed! did not get 200 response for last URL.')
        continue;
    
