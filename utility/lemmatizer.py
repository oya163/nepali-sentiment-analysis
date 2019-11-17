'''
	Lemmatize all the text files into one file
        Uses morfessor
	
	Author - Oyesh Mann Singh
	Date - 10/23/2016
	
	How to run:
		python utility/lemmatizer.py
'''

import os
import argparse
import morfessor

parser = argparse.ArgumentParser(add_help=True, description=('Text Collector Parser'))
parser.add_argument('--input_dir', '-idir', default='./data/youtube/sampled/txt', metavar='PATH', help='Input path directory')
parser.add_argument('--output_dir', '-odir', default='./data/youtube/sampled/txt', metavar='PATH', help='Output path directory')
parser.add_argument('--unicode', '-u', default='utf8', choices=['utf8','utf16'], metavar='UTF', help='Encoding format')

args = parser.parse_args()


def main():
    input_dir = args.input_dir
    output_dir = args.output_dir
    encoding = args.unicode
    

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            input_file = os.path.join(root, f)
            outname = os.path.basename(f.split('.')[0])+'.lemma'
            output_file = os.path.join(root, outname)
            print(output_file)
            #i_f = open(filename, 'r', encoding=encoding)
            #reader = i_f.readlines()
            with open(output_file, 'w', encoding=encoding) as outfile:
                cmd = "morfessor-segment -L data/morpheme/morpheme.sgm {0} > {1}".format(input_file, output_file)
                os.system(cmd)
                


if __name__ == "__main__":
	main()
