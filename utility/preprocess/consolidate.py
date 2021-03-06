'''
	Consolidate all CoNLL files
	into one CoNLL file for each video
	
	Author - Oyesh Mann Singh
	Date - 01/15/2020
	
	How to run:
		python utility/conlltodataset.py
        
    Output Directory structure:
       |_ Avenues_khabar
          |_ 0S8tX4eRa6M.conll
          |_ 3XI16CXFcJA.conll
          |_ .
          |_ .
       |_ Canada_Nepal
          |_ 7aSSjVrkYmI.conll
          |_ 7I4HfcAnYzA.conll
          |_ .
          |_ .
'''

import os
import argparse
import subprocess
import shutil

parser = argparse.ArgumentParser(add_help=True, description=('CoNLL to Final Dataset'))
parser.add_argument('--input_dir', '-idir', default='../brat/data/annotated/', metavar='PATH', help='Input path directory')
parser.add_argument('--output_dir', '-odir', default='./data/dataset', metavar='PATH', help='Output path directory')
parser.add_argument('--unicode', '-u', default='utf8', choices=['utf8','utf16'], metavar='UTF', help='Encoding format')

args = parser.parse_args()


def main():
    input_dir = args.input_dir
    output_dir = args.output_dir
    encoding = args.unicode
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
#   Delete channel directory in output folder if exists
    for channels in next(os.walk(input_dir))[1]:
        output_folder = os.path.join(output_dir, channels)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            
    # Go through each folder 
    # Create a conll file based on video name
    total_file = os.path.join(output_dir, 'total.conll')
    with open(total_file, 'w', encoding=encoding) as t_f:
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                filename = os.path.join(root, f)
                if filename.split('.')[-1] == 'conll':
                    pathlist = filename.split('/')
                    channel_name = pathlist[-3]
                    output_folder = os.path.join(output_dir, channel_name)

                    if not os.path.exists(output_folder):
                        os.mkdir(output_folder)

                    video_name = pathlist[-2]
                    output_file = os.path.join(output_folder, video_name + '.conll')

                    with open(output_file, 'a', encoding=encoding) as o_f:
                        i_f = open(filename, 'r', encoding=encoding)
                        data = i_f.read()
                        o_f.write(data)
                        t_f.write(data)
                        o_f.write('\n')
                        t_f.write('\n')


if __name__ == "__main__":
	main()

