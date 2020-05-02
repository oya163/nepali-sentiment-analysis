#!/usr/bin/env python

'''
    Sample the data from a larger dataset
    Author : Sandesh Timilsina
    Date: 10/11/2019

    How to run:
    python <filename.py> -idir <input_dir> -odir <dutput_dir> -s <sample-size>
    python sampler.py -idir ../data/youtube/raw -odir ../data/youtube/sampled/
    python sampler.py

    Description:
    - Sample out n comments based on the number of likes from the dataset
    - Creates a json file of n comments
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
'''
    Removes all emojis completely for text file
    Because text files are used in BRAT annotation
    Emojis are not necessary as of now
    Also, causing problem in lemmatizer
'''
def write_txt(output_dir, sample_data):
    for obj in sample_data:
        cid = obj['id']
        filename = os.path.join(output_dir, cid+'.txt')
       
        comment = obj['text']

        # Removal of emoji
        plain_txt = emoji_pattern.sub(r' ', comment).split()

        # If sentence ender not found, then add it
        # Because we need to split sentence in unitag.exe
        if plain_txt[-1] not in ['?', '!', '|', '।'] and plain_txt[-1][-1] not in ['?', '!', '|', '।']:
            plain_txt.append('।')

        # Cite the paper
        # Removal of very short or lengthy comments
        if len(plain_txt) > 5 and len(plain_txt) < 50:
            with open(filename, 'w', encoding='utf-16') as f:
                f.write(' '.join(plain_txt)+"\n")
            f.close()


# Randomly sample comments
def process_file(file, input_dir, size):
    data = json.load(open(os.path.join(input_dir, file),'r',encoding='utf-8'))
    json_data = []
    for item in data:
        comment = item['text']
        if (check_devnagari(comment)):
            json_data.append(item)
    json_data.sort(key= lambda k : k['likes'], reverse=True)
    return json_data[:100]


# Store into given folder
def process_folder(input_dir, output_dir_json, output_dir_txt, size):
    for filename in os.listdir(input_dir):
        print(os.path.join(input_dir,filename))
        file_name = filename.split('.')
        if filename.endswith(".json"):
            sample_data = process_file(filename, input_dir, size)
            write_json(output_dir_json, filename, sample_data)
            txt_dir = os.path.join(output_dir_txt, file_name[0])
            if not os.path.exists(txt_dir):
                os.mkdir(txt_dir)
            write_txt(txt_dir, sample_data)
        

# Process arguments
def main(argv):
    parser = argparse.ArgumentParser(add_help=True, description=('Random Sample Parser'))
    parser.add_argument('--directory', '-idir', default='./data/youtube/raw', metavar='PATH', help='Input folder directory')
    parser.add_argument('--output_dir', '-odir', default='./data/youtube/sampled', metavar='PATH', help='Output folder directory')
    parser.add_argument('--sample', '-s', metavar='N', default=100, type=int, help='Sample size <default:100>')

    args = parser.parse_args(argv)
    input_dir = args.directory
    output_dir = args.output_dir
    size = args.sample

    if not os.path.exists(input_dir):
        raise ValueError('Input Folder not found')


    for dirs in os.listdir(input_dir):
        input_path = os.path.join(input_dir, dirs)
        root_dir = os.path.basename(input_path)
        output_dir_json = os.path.join(output_dir, "json", root_dir)
        output_dir_txt = os.path.join(output_dir, "txt", root_dir)

        if not os.path.exists(output_dir_json):
            os.makedirs(output_dir_json)
        if not os.path.exists(output_dir_txt):
            os.makedirs(output_dir_txt)    

        process_folder(input_path, output_dir_json, output_dir_txt, size)
        

if __name__ == "__main__":
    main(sys.argv[1:])
