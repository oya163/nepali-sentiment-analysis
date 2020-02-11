#!/usr/bin/env python

'''
    Collects all the comments
    Creates social media corpus
    
    Author : Oyesh Mann Singh
    Date: 12/27/2019

    How to run:
    python corpus_creator.py -i ../data/corpus/all_comments

    Description:
    - Collects all the comments from list of videos id
      given in a input_file.txt
'''

import os
import sys
import re
import json
import argparse
import numpy as np


# Read each comments from given json file
def process_file(jsonfile):
    try:
        data = json.load(open(jsonfile, 'r', encoding='utf8'))
        filename = os.path.splitext(jsonfile)[0] + '.txt'
        with open(filename, 'w+') as f:
            for obj in data:
                comment = obj['text']
                f.write(comment+"\n")
        f.close()        
    except ValueError:
        print("Empty JSON!!!")


# Read each json file
def process_folder(input_dir):
    for filename in os.listdir(input_dir):
        jsonfile = os.path.join(input_dir, filename)
        if jsonfile.endswith(".json"):
            print(jsonfile)
            process_file(jsonfile)

# Process arguments
def main(argv):
    parser = argparse.ArgumentParser(add_help=True, description=('Corpus Creator Parser'))
    parser.add_argument('--ifile', '-i', default='../data/corpus/all_comments', metavar='PATH', help='Input folder directory')

    args = parser.parse_args(argv)
    input_dir = args.ifile

    if not os.path.exists(input_dir):
        raise ValueError('Input Folder not found')

    process_folder(input_dir)
        

if __name__ == "__main__":
    main(sys.argv[1:])
