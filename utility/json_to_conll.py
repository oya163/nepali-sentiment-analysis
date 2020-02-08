'''
	Converts the .json file to .conll format
	Author: Sandesh Timilsina
	Date: 02/07/2020
	
	How to run:
	- 
'''

import os
import argparse
import pandas as pd
import json

def read_jsonfile(file):
    with open(file) as f:
        data = json.load(f)
    return data

def re_label(comment, item, df):
    tags = item['tags']
    for tag in tags:
        if 'entity' in tag:
            entity = tag['entity'].split()
            entity_cat = tag['entity_cat']
            begin = True
            for item in entity:
                begin=True
                idx = df[df[0]==item].index.tolist()
                for i in range(len(idx)):
                    if idx[i]>=0 and df.loc[idx[i], 1]=='O':
                        if begin:
                            df.loc[idx[i], 1]= 'B-'+entity_cat
                            begin= False
                        else:
                            df.loc[idx[i], 1]= 'I-'+entity_cat
                        
        if 'aspect' in tag:
            aspect = tag['aspect'].split()
            aspect_cat = tag['aspect_cat']
            begin = True
            for item in aspect:
                idx = df[df[0]==item].index.tolist()
                for i in range(len(idx)):
                    if idx[i]>=0 and df.loc[idx[i], 1]=='O':
                        if idx[i]>=0:
                            df.loc[idx[i], 1]= 'B-'+aspect_cat
                            begin = False
                        else:
                            df.loc[idx[i], 1]= 'I-'+aspect_cat
    return df


def text_to_conll(data):
    df = None
    final_df = pd.DataFrame()
    count=0
    for item in data:
        count+=1
        df = pd.DataFrame()
        comment = item['comment']
        for word in comment.split():
            df = df.append(pd.Series([word, 'O']), ignore_index=True)
        final_df = re_label(comment, item, df)
        final_df = final_df.append(pd.Series(["",""]), ignore_index=True)
        final_df.to_csv('/home/sandesh/Desktop/sand.conll', header=None, index=None, sep='\t', mode='a')
    print('Done! ',count,' comments converted to conll format')

def json_to_conll(file_path):
    data = read_jsonfile(file_path)
    text_to_conll(data)

