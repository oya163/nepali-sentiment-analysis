'''
	Converts the .json file to .conll format
	Author: Sandesh Timilsina
	Date: 02/07/2019
	
	How to run:
	- Pass the all_channels.json file as command line argument.
    Example:
        >> python json_to_conll.py -file ../../brat/data/nepsa/all_channels.json
'''

import os
import sys
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


def text_to_conll(data, file_path):
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
        output_file = file_path.split('.')[0]+'.conll'
        final_df.to_csv(output_file, header=None, index=None, sep='\t', mode='a')
    print('Done! ',count,' comments converted to conll format')


def json_to_conll(file_path):
    data = read_jsonfile(file_path)
    text_to_conll(data, file_path)


def main(argv):
    parser = argparse.ArgumentParser(add_help=True, description=('Json to Conll converter'))
    parser.add_argument('-file', default='../../data/nepsa/al_channels.json', help='Input json file')
    
    args = parser.parse_args(argv)
    input_file = args.file
    
    if not os.path.exists(input_file) or not input_file.endswith(".json"):
        raise ValueError('Invalid Input Folder')

    json_to_conll(input_file)    
    print(' Successfully completed')
        

if __name__ == "__main__":
    main(sys.argv[1:])
    