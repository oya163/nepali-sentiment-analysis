#!/usr/bin/env python3
'''
    Merge two conll files to check F1 and Kappa-score
    for inter-annotator aggrement
    
    Author: Oyesh Mann Singh
    Date: 11/12/2019

'''

import os
import argparse
import pandas as pd
import numpy as np
import csv
import shutil

from sklearn.metrics import cohen_kappa_score as kappa

try:
    import conll_eval as e
    import utilities as utilities
except ImportError:
    import utility.conll_eval as e
    import utility.utilities as utilities

    
def merger(input_dir_1, input_dir_2, output_file, verbose):
    """
        Merges CoNLL files from two annotators.
        
        Parameters
        ----------
        input_dir_1 : directory
            Input directory path from first annotator
        input_dir_2 : directory
            Input directory path from second annotator
        output_file : output file name
            Output file name after merge
        verbose : boolean
            Whether to print out which files are being processed
            
        Returns
        -------
        a1_tokens : list
            List of tokens from first annotator
        a2_tokens : list
            List of tokens from second annotator
        
    """
    input_files_1 = os.listdir(input_dir_1)
    input_files_2 = os.listdir(input_dir_2)
    
    common_files = set(input_files_1).intersection(set(input_files_2))
    a1_tokens = []
    a2_tokens = []
    with open(output_file,'w', encoding='utf-8') as out_f:
        for each in common_files:
            input_file_1 = os.path.join(input_dir_1, each)
            input_file_2 = os.path.join(input_dir_2, each)
            if verbose:
                print("Processing file 1:", input_file_1)
                print("Processing file 2:", input_file_2)

            with open(input_file_1,'r', encoding='utf-8') as first_file, open(input_file_2,'r', encoding='utf-8') as second_file:
                for (i1,row1),(i2,row2) in zip(enumerate(first_file), enumerate(second_file)):
                    #To know which line is defunct in file
                    #print(i+1)
                    if len(row1) > 1 or len(row2) > 1:
                        row1 = row1.split()
                        row2 = row2.split()
                        token = row1[3]
                        gt = row1[0]
                        pt = row2[0]
                        a1_tokens.append(gt)
                        a2_tokens.append(pt)
                        out_f.write(token+'\t'+gt+'\t'+pt+'\n')
                    else:
                        out_f.write('\n')
            out_f.write('\n')
            
    return a1_tokens, a2_tokens
        
        
def main(**args):
    input_dir_1 = args["input_dir_1"]
    input_dir_2 = args["input_dir_2"]
    output_file = args["output_file"]
    log_file = args["log_file"]
    verbose = args["verbose"]
    
    # Remove old output file
    if os.path.exists(output_file):
        os.remove(output_file)

    # Remove old log file
    if os.path.exists(output_file):
        os.remove(output_file)        
    
    # Start merging conll
    print("***************Merging Files***************")    
    a1_tokens, a2_tokens = merger(input_dir_1, input_dir_2, output_file, verbose)
    
    # Start evaluating merged conll file
    logger = utilities.get_logger(log_file)
    print("***************F1 Evaluation Metric***************")
    e.evaluate_conll_file(logger, output_file)
    
    print("***************Kappa Evaluation Metric***************")
    logger.info("Kappa coefficient = {}".format(kappa(a1_tokens, a2_tokens)))
    

if __name__=="__main__":
    parser = argparse.ArgumentParser("CoNLL file Merge Argument Parser")
    parser.add_argument("-i", "--input_dir_1", default="./test/test_1", metavar="PATH", help="Input first dir path")
    parser.add_argument("-j", "--input_dir_2", default="./test/test_2", metavar="PATH", help="Input second dir path")
    parser.add_argument("-o", "--output_file", default="./test/merged.conll", metavar="FILE", help="Output File Name")
    parser.add_argument("-l", "--log_file", default="./logs/annotation.log", metavar="FILE", help="Log File Name")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, help="Print description")

    args = vars(parser.parse_args())

    main(**args)
