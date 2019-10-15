#!/usr/bin/env python

'''
    Sample the data from a larger dataset
    Author : Sandesh Timilsina
    Date: 10/11/2019

    How to run:
    python <filename.py> -idir <input_dir> -odir <dutput_dir> -s <sample-size>

    Description:
    - Sample out n data from the dataset
    - Creates a json file of n data
    - Creates a txt file with comments only
    - Deemojify the comment
'''

import os
import sys
import re
import json
import argparse
import numpy as np
import emoji

# Regex emoji symbols
# https://stackoverflow.com/a/33417311
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


# Check if the given comment has Devanagari unicode
def check_devnagari(comment):
    count = re.findall("[\u0900-\u097F]+", comment)
    if (len(count)>0):
        return True
    return False


# Write into JSON file
def write_json(output_dir, file, sample_data):
    with open(os.path.join(output_dir, file), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)


# Write into TXT file
def write_txt(output_dir, file, sample_data):
    with open(os.path.join(output_dir, file), 'w', encoding='utf-16') as f:
        for obj in sample_data:
            comment = obj['text'];
            #plain_txt = emoji.demojize(comment)
            plain_txt = emoji_pattern.sub(r'', comment)
            f.write(plain_txt+"\n")


# Randomly sample comments
def process_file(file, input_dir, size):
    data = json.load(open(os.path.join(input_dir, file),'r',encoding='utf-8'))
    json_data = []
    for item in data:
        comment = item['text']
        if (check_devnagari(comment)):
            json_data.append(item)
    if (len(json_data)>size):
        return np.random.choice(json_data, size, replace=False).tolist()
    return json_data


# Store into given folder
def process_folder(input_dir, output_dir_json, output_dir_txt, size):
    for filename in os.listdir(input_dir):
        print(filename)
        if filename.endswith(".json"):
            sample_data = process_file(filename, input_dir, size)
            write_json(output_dir_json,filename,sample_data)
            write_txt(output_dir_txt,filename.split(".")[0]+".txt",sample_data)
        

# Process arguments
def main(argv):
    parser = argparse.ArgumentParser(add_help=True, description=('Random Sample Parser'))
    parser.add_argument('--directory', '-idir', metavar='PATH', help='Input folder directory')
    parser.add_argument('--output_dir', '-odir', metavar='PATH', help='Output folder directory')
    parser.add_argument('--sample', '-s', metavar='N', default=100, type=int, help='Sample size <default:100>')
    try:
        args = parser.parse_args(argv)
        input_dir = args.directory
        output_dir = args.output_dir
        size = args.sample

        if not os.path.exists(input_dir):
            raise ValueError('Input Folder not found')
        
        output_dir_json = output_dir+"_json"
        output_dir_txt = output_dir+"_txt"
        
        if not os.path.exists(output_dir_json):
            os.makedirs(output_dir_json)
        if not os.path.exists(output_dir_txt):
            os.makedirs(output_dir_txt)    
        
        process_folder(input_dir, output_dir_json,output_dir_txt, size)
        
    except Exception as e:
        print('Error: ',str(e))
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
