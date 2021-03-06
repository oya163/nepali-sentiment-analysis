'''
	Collect all the text files
	into one file
	
	Author - Oyesh Mann Singh
	Date - 10/17/2019
	
	How to run:
		python utility/text_collector.py
'''

import os
import argparse

parser = argparse.ArgumentParser(add_help=True, description=('Text Collector Parser'))
parser.add_argument('--input_dir', '-idir', default='./data/youtube/sampled/txt', metavar='PATH', help='Input path directory')
parser.add_argument('--output_file', '-ofile', default='./data/corpus/youtube_corpus.txt', metavar='PATH', help='Output file')
parser.add_argument('--unicode', '-u', default='utf16', choices=['utf8','utf16'], metavar='UTF', help='Encoding format')

args = parser.parse_args()


def main():
    input_dir = args.input_dir
    output_file = args.output_file
    encoding = args.unicode
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                filename = os.path.join(root, f)
                i_f = open(filename, 'r', encoding=encoding)
                reader = i_f.readlines()
                for row in reader:
                    outfile.write(row)


if __name__ == "__main__":
	main()

