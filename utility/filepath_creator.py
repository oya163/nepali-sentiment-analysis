'''
	Stemmer + POS tagger based on 
	https://www.lancaster.ac.uk/staff/hardiea/nepali/postag.php
	
	Better to run unitag separately
	because recurring questions for overwriting files
	
	Just create path for each text files from sampled data
    Helper for unitag.exe
    Unfortunately, unitag.exe can be executed from Windows only
    It is very old program.
    Work in progress for new lemmatizer
    
    Author: Oyesh Mann Singh
    Date: 10/16/2019
    
    How to run:
        python filepath_creator.py --idir <Input Directory> --ofile <filename.txt>
        or,
        python filepath_creator.py --idir ./data/youtube/sampled --ofile ./nepali-unitag/bin/filelist.txt
        or,
        python utility/filepath_creator.py
'''

import sys
import os
import argparse

parser = argparse.ArgumentParser(add_help=True, description=('File Path Creator Parser'))
parser.add_argument('--input_dir', '-idir', default='./data/youtube/sampled', metavar='PATH', help='Input folder directory')
parser.add_argument('--output_file', '-ofile', default='./nepali-unitag/bin/filelist.txt', metavar='PATH', help='Output filepath')

args = parser.parse_args()

filelist = args.output_file
dir = args.input_dir

def main():
    if os.path.exists(filelist):
        os.remove(filelist)

    file_count = 0
    with open(filelist, 'w', encoding='utf-8') as flist:
        for root, dirs, files in os.walk(dir):
            for f in files:
                input_file = os.path.join(root, f)
                print(input_file)
                file_count += 1
                flist.write(input_file+'\n')
                
    print("Number of files to be processed: ", file_count)

                
if __name__ == "__main__":
    main()
    