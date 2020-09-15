#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:59:53 2019

@author: ansh
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

url_file = 'prNW-scrapped-URLs-500.csv'

BASE_URLS = ['https://www.prnewswire.com/news-releases/financial-services-latest-news/contracts-list/',
             'https://www.prnewswire.com/news-releases/policy-public-interest-latest-news/labor-union-news-list/',
             'https://www.prnewswire.com/news-releases/financial-services-latest-news/joint-ventures-list/',
             'https://www.prnewswire.com/news-releases/policy-public-interest-latest-news/fda-approval-list/',
             'https://www.prnewswire.com/news-releases/financial-services-latest-news/dividends-list/',
             ]


URL_SCRAPED = {} # dictionary  { topic_name : [list of URLs] }

for base_url in BASE_URLS:
    links = []
    for i in range(5): # cap to 500 urls, each page provides 100 urls
        seed_url = base_url+'?page='+str(i+1)+'&pagesize=100'
        print('fetching urls topic urls from: ' + seed_url)
        response = requests.get(seed_url)

        if (response.status_code == 200):
            soup = BeautifulSoup(response.text, "html.parser")
            temp_links = soup.find_all('a', {'class':'news-release'})
            if temp_links.__len__() == 0: break;
            links.extend(temp_links)
            time.sleep(1)
        else:
            print('did not receive 200 response at page: ', i+1)
            print('fetched sub_links from {} pages using base url : {}'.format(i, base_url))
            break;
    
    URL_SCRAPED[base_url.split('/')[-2][:-5]] = links



print('url scraping completed...!!!\nSUMMARY:')

for topic in URL_SCRAPED:
    print('fetched {} sub_links for topic {}'.format(URL_SCRAPED[topic].__len__(), topic))


#create dataframe from dictionary, and save in csv
#########################################################################################

df = pd.DataFrame(columns=['topics', 'URL', 'document_title'])
for topic in URL_SCRAPED:
    for i in range(len(URL_SCRAPED[topic])):
        df = df.append({'topics':topic, 'URL':URL_SCRAPED[topic][i]}, ignore_index=True)


#grab document title 
doc_title = [x.string for x in df.URL]

df.URL = ['https://www.prnewswire.com'+x.get('href') for x in df.URL]
df.document_title = doc_title


df.to_csv(url_file, index=False)

print('saved all urls in csv file', url_file)


