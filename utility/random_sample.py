'''
    Sample the data from a larger dataset
    Author : Sandesh Timilsina
    Date: 10/11/2019

    How to run:
    python <filename.py> -idir <input_dir> -odir <dutput_dir> -s <sample-size>
'''
import os
import sys
import re
import json
import argparse
import numpy as np


def check_devnagari(comment):
    count = re.findall("[\u0900-\u097F]+", comment)
    if (len(count)>0):
        return True
    return False

def process_file(file,input_dir,output_dir,size):
    data = json.load(open(os.path.join(input_dir, file),'r',encoding='utf8'))
    json_data = []
    for item in data:
        comment = item['text']
        if (check_devnagari(comment)):
            json_data.append(item)
    print("size",size)
    sample_data = np.random.choice(json_data, size, replace=False)
    print(sample_data)
    with open(os.path.join(output_dir, file), 'w', encoding='utf-8') as f:
        json.dump(sample_data.tolist(), f, ensure_ascii=False, indent=4)

def process_folder(input_dir,output_dir,size):
    for filename in os.listdir(input_dir):
        print(filename)
        if filename.endswith(".json"):
            process_file(filename, input_dir, output_dir,size)
                
        
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

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        process_folder(input_dir,output_dir,size)
        
    except Exception as e:
        print('Error: ',str(e))
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
