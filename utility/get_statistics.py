#!/usr/bin/env python3

# Simple program to check data statistics of NER file
# Input file should be in standard Stanford format
# Outputs number of PER, LOC, ORG tags

import csv
import argparse
import os
from collections import Counter
from tabulate import tabulate

def main():
	parser = argparse.ArgumentParser(description='Input file name')
	
	parser.add_argument('filename', metavar='STR',
                    help='input valid file path')
					
	args = parser.parse_args()
	
	input_file = args.filename
	
	sent_counter = 0
    
	entities = {}
	high_entities = {}    
    
	with open(input_file,'r', encoding='utf-8') as in_file:
		reader = csv.reader(in_file, delimiter='\t')
		for row in reader:
			if row:
				if row[-1] not in entities:
					entities[row[-1]] = 1
				else:
					entities[row[-1]] += 1
                    
				if row[-1][2:] not in high_entities:
					high_entities[row[-1][2:]] = 1
				else:
					high_entities[row[-1][2:]] += 1                    
			else:
				sent_counter += 1

	print("***********Detailed statistics of entities***********")        
	headers = ['Entities', 'Count']
	data = sorted([(k,v) for k,v in entities.items()]) # flip the code and name and sort
	print(tabulate(data, headers=headers))

	print("***********High level statistics of entities***********")
	data = sorted([(k,v) for k,v in high_entities.items()]) # flip the code and name and sort
	print(tabulate(data, headers=headers))
    
	print("********************************************")
	print("Total count of sentences = ", sent_counter)
    
    
if __name__ == "__main__":
	main()
