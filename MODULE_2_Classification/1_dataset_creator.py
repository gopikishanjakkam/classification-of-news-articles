#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 00:38:24 2019

@author: ansh
"""
import random
import pandas as pd
from glob import glob


data_path = '../scrapped_data'
max_doc = 50 # specify maximum limit on # documents in each category
shuffle = True # shuffle data from different sources



def read_doc_path(data_path, shuffle=True):
    '''
    this function returns the file path of each document of each class in a dictionary.
    
    shuffles the file path from different sources if shuffle=True
    data_path = path_of_data_directory
    a document is situated in the following directory sturcture:
        data_path/[document_source]/[class_name]/[document.txt]
    '''
    file_path = {}
    for source in glob(data_path+'/*'):
        for category_dir in glob(source+'/*'):
            category_name = category_dir.split('/')[-1]
            print('getting file_paths from: ' + category_dir)
            if category_name in file_path.keys():
                file_path[category_name].extend(glob(category_dir+'/*'))
            else:
                file_path[category_name] = glob(category_dir+'/*') #list of document path in this class

    for item in file_path:
        print("obtained {} document's file path for class {}".format(len(file_path[item]), item))
    
    # shuffle the documents from  different sources
    # for classes with documents with multiple sources,
    if(shuffle):
        for item in file_path: random.shuffle(file_path[item])
    
    return file_path



def create_dataset(file_path, max_doc=None, save_filename='./dataset.csv'):
    '''
    create a single csv file of document text and labels
    
    file_path : dict of {class_name : [list of document file paths]}
    max_docs : (int) sets a limit to number of document read from file_path for each class
                if max_doc = None, reads all documents from all class
    '''
    
    class_name = []
    headline = []
    situation = []
    for item in file_path:
        path_list = file_path[item][:max_doc]
        for doc_path in path_list:
            with open(doc_path, 'r') as f:
                text = f.read()
                headline.append(text[:text.find('SITUATION')])
                situation.append(text[text.find('SITUATION'):])
                class_name.append(item)
    df = pd.DataFrame({'label': class_name, 'headline':headline, 'situation':situation})
    df.to_csv(save_filename, index=False)







file_path = read_doc_path(data_path, shuffle=False)
create_dataset(file_path, max_doc=10, save_filename='../dataset/dataset_test_10.csv')
#create_dataset(file_path, max_doc=200, save_filename='../dataset/dataset_small_200.csv')
create_dataset(file_path, max_doc=None, save_filename='../dataset/dataset_large_all.csv')




















