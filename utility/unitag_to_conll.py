'''
	Convert the output of unitag.exe
	into CoNLL format file
	
	Author - Oyesh Mann Singh
	Date - 10/16/2016
	
	How to run:
		python unitag_to_conll.py -idir ..\data\youtube\sampled\txt
'''

import csv
import sys
import os
import argparse
import shutil

parser = argparse.ArgumentParser(add_help=True, description=('Unitag format to CoNLL Format Parser'))
parser.add_argument('--input_dir', '-idir', metavar='PATH', help='Input path directory')

args = parser.parse_args()

# Converts Unitag.exe output file
# into CoNLL format file
def converter(inputfile, outputfile):
	with open(inputfile,'r', encoding='utf16') as in_file, open(outputfile,'w', encoding='utf8') as out_file:
		reader = csv.reader(in_file, delimiter='\t', quoting=csv.QUOTE_NONE, skipinitialspace=True)
		for row in reader:
			curr_sent, word_num, word = row[0].split()
			tag, pos_tag = row[1].split()
			pos_tag = pos_tag.split('/')[0]
			if pos_tag == 'YF':
				out_file.write(word+'\t'+pos_tag+'\n')
				out_file.write('\n')
			else:
				out_file.write(word+'\t'+pos_tag+'\n')
				
				
# Extracts text_only and tag_only from
# CoNLL format file
def text_tag(inputfile, sent_file, tag_file):
	with open(inputfile,'r', encoding='utf-8') as in_file, open(sent_file,'w', encoding='utf-8') as txt_f, open(tag_file,'w', encoding='utf-8') as tag_f:
		sentence = []
		tag = []
		for i,row in enumerate(in_file):
			# To know which line is defunct in file
	#         print(i+1)
			if len(row)>2:
				row = row.strip().split("\t")
				sentence.append(row[0])
				tag.append(row[1])                
			else:
				if len(sentence) > 3:
					txt_f.write(' '.join(sentence)+'\n')
					tag_f.write(' '.join(tag)+'\n')
				
				sentence = []
				tag = []
				
	in_file.close()
	txt_f.close()
	tag_f.close()
	print("Text tag file prepared !!!")
				
				
def main():
	for root, dirs, files in os.walk(args.input_dir):
		for file in files:
			filename = file.split('.')
			if filename[1] == 'txt_utg':
				input_file = os.path.join(root, file)	
				out_file = os.path.join(root, filename[0]+'.iob.txt')
				
				# Converts Unitag format to CoNLL format 
				print("Processing file : ", input_file)
				converter(input_file, out_file)
				
				sent_file = os.path.join(root, filename[0]+'.sent.txt')
				tag_file = os.path.join(root, filename[0]+'.tag.txt')
				
				# Creates text_only and tag_only files
				print("Processing file : ", out_file)
				text_tag(out_file, sent_file, tag_file)

				
if __name__ == "__main__":
	main()