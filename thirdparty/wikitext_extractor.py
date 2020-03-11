#!/usr/bin/env python3

'''
    Wiki Text Extractor file
    Author: Oyesh Mann Singh
    
    How to run:
        ./main.py -i <input_dir> -o <output_dir>
        
    Note:
    <input_dir> = contains all the files in json format
    
    The files should be extracted using Wikiextractor.py
    https://github.com/attardi/wikiextractor/blob/master/WikiExtractor.py
'''

import os
import json
import argparse
from pathlib import Path

def parse_args():
    """
        Argument Parser
    """
    parser = argparse.ArgumentParser(description="Wiki Text Extractor")
    
    parser.add_argument("-i", "--input_dir", dest="input_dir", type=str, metavar="PATH", 
                        default="./extracted",help="Input directory path ")
    parser.add_argument("-o", "--output_dir", dest="output_dir", type=str, metavar="PATH", 
                        default="./wiki_text",help="Output directory path")
    parser.add_argument("-t", "--output_type", dest="output_type", type=int, metavar="INT", 
                        default=1, choices=[1,2], help="Output in a single file or multiple file")     
    args = parser.parse_args()
    return args

def multi_file(args):
    input_dir = args.input_dir
    root_dir = args.output_dir
    
    paths = [str(x) for x in Path(input_dir).glob("**/wiki_*")]
    
    for each in paths:
        print("Processing : ", each)
        input_dir = each.split('/')

        output_dir = os.path.join(root_dir, input_dir[1])
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        o_file = input_dir[-1]+'.txt'
        output_file = os.path.join(output_dir, o_file)

        with open(each, 'r') as in_f, open(output_file, 'w') as out_f:
            for line in in_f.readlines():
                j = json.loads(line)
                out_f.write(j["text"])    
    
def single_file(args):
    input_dir = args.input_dir
    output_file = os.path.join(args.output_dir, 'wiki_text.txt')
    
    paths = [str(x) for x in Path(input_dir).glob("**/wiki_*")]
    
    with open(output_file, 'w') as out_f:
        for each in paths:
            print("Processing : ", each)
            with open(each, 'r') as in_f:
                for line in in_f.readlines():
                    j = json.loads(line)
                    out_f.write(j["text"])  
                
def main():
    args = parse_args()
    if args.output_type == 1:
        # Dump everything into one single file
        single_file(args)
    else:
        # Dump everything into multiple files as input
        multi_file(args)
          
        
if __name__ == "__main__":
    main()