#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 11:40:06 2019

@author: ansh
"""
import pandas as pd
from bs4 import BeautifulSoup
import os
import re


TOPIC = []
URL = []
TITLE = []

webpage_dir = '../globalNW-webpage-scrape/'
file_list = os.popen('ls '+ webpage_dir).read().split('\n')[:-1]


for wp in file_list:
    
    f = open(webpage_dir+wp, 'r').read()
    print('extracting URLs from webpage: ', wp)
    soup = BeautifulSoup(f, "html.parser")
    links = soup.findAll('a', attrs={'href': re.compile("^/news-release")})

    if len(links) == 0:
        links = soup.findAll('a', attrs={'href': re.compile(".html$")})
        urls = [x.get('href') for x in links]
    else:
        urls = ['https://www.globenewswire.com'+x.get('href') for x in links]
    titles = [x.string for x in links]
    topics = [wp.split('-')[0]] * len(urls)
    
    TOPIC.extend(topics)
    URL.extend(urls)
    TITLE.extend(titles)
    



df = pd.DataFrame({'topic':TOPIC, 'URL':URL, 'title':TITLE})
df.to_csv('globalNW-scrapped-URLs.csv', index=False)