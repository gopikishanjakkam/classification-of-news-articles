#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:15:26 2019

@author: ansh
"""

import os
import time
import requests
from os import path
import pandas as pd
from bs4 import BeautifulSoup


SAVE_PATH = '../../scrapped_data/GLOBAL_data/'
START_RANGE = 2460
END_RANGE = -1

url_file = '../globalNW-scrapped-URLs.csv'
url_df = pd.read_csv(url_file)


for row in url_df.iloc[START_RANGE : END_RANGE].iterrows():
#for row in url_df.iterrows():
    metadata = row[1]
    topic_dir = SAVE_PATH + metadata.topic + '/'
    filename = topic_dir + str(row[0]) + '.txt'
    
    #skip if file already exists
    if (path.exists(filename)):
        print(' data from url {} exists, skipping!'.format(metadata.URL))
        continue;
    time.sleep(1)
    response = requests.get(metadata.URL+'?print=1') 
    if (response.status_code == 200):
        soup = BeautifulSoup(response.text, "html.parser")  
        try:
            doc_title = soup.find('h2').string
            doc_text = soup.find_all(['p', 'tr']) # fixed code to extract text from tables as well.
            doc_text = ''.join([x.text for x in doc_text])
        except:
            print('ERROR parsing retrieved html! doc index:{}'.format(row[0]))
            continue;
        if not os.path.exists(topic_dir): os.system('mkdir -p ' + topic_dir)
        
        print('FETCHED data from url {}, SAVING in file {}'.format(metadata.URL, filename))
        
        with open(filename, 'w') as f:
            f.write('HEADLINE : {}\n'.format(doc_title))
            f.write('SITUATION : \n' + doc_text)
    else:
        print('fetching failed! did not get 200 response for last URL.')
        continue;

