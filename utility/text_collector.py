'''
	Collect all the text files
	into one file
	
	Author - Oyesh Mann Singh
	Date - 10/17/2016
	
	How to run:
		python text_collector.py -idir ..\data\youtube\sampled\txt -o output_file.txt
'''

import os
import argparse

parser = argparse.ArgumentParser(add_help=True, description=('Unitag format to CoNLL Format Parser'))
parser.add_argument('--input_dir', '-idir', metavar='PATH', help='Input path directory')
parser.add_argument('--output_file', '-o', metavar='PATH', help='Output file')

args = parser.parse_args()


def main():
	input_dir = args.input_dir
	output_file = args.output_file

	with open(output_file, 'w', encoding='utf-8') as o_f:
		for file in os.listdir(input_dir):
			filename = os.path.join(input_dir, file)
			i_f = open(filename, 'r', encoding='utf-16')
			reader = i_f.readlines()
			for row in reader:
				o_f.write(row)
	o_f.close()


				
if __name__ == "__main__":
	main()

