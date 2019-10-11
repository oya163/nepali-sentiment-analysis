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


def check_devnagari(comment):
    count = re.findall("[\u0900-\u097F]+", comment)
    if (len(count)>0):
        return True
    return False

def write_json(output_dir, file, sample_data):
    with open(os.path.join(output_dir, file), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)

def write_txt(output_dir, file, sample_data):
    with open(os.path.join(output_dir, file), 'w', encoding='utf-16') as f:
        for obj in sample_data:
            comment = obj['text'];
            plain_txt = emoji.demojize(comment)
            f.write(plain_txt+"\n")

def process_file(file,input_dir,size):
    data = json.load(open(os.path.join(input_dir, file),'r',encoding='utf-8'))
    json_data = []
    for item in data:
        comment = item['text']
        if (check_devnagari(comment)):
            json_data.append(item)
    if (len(json_data)>size):
        return np.random.choice(json_data, size, replace=False).tolist()
    return json_data

def process_folder(input_dir,output_dir_json,output_dir_txt,size):
    for filename in os.listdir(input_dir):
        print(filename)
        if filename.endswith(".json"):
            sample_data = process_file(filename, input_dir, size)
            write_json(output_dir_json,filename,sample_data)
            write_txt(output_dir_txt,filename.split(".")[0]+".txt",sample_data)
        
def main(argv):
    parser = argparse.ArgumentParser(add_help=True,description=('Same rows from the dataset'))
    parser.add_argument('--directory', '-idir', help='Input folder directory')
    parser.add_argument('--output_dir', '-odir', help='Output folder directory')
    parser.add_argument('--sample', '-s', type=int, help='Sample size')
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
