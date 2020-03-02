#!/usr/bin/env python3
'''
    Merge two conll files to check F1 and Kappa-score
    for inter-annotator aggrement
    
    Author: Oyesh Mann Singh
    Date: 11/12/2019

    How to run:
        python utility/agreement.py -i test/oyesh_conll_v4/ -j test/sandesh_conll_v4/ -v
        
        python utility/agreement.py -i data/agreement/total_oyesh.conll -j data/agreement/total_sandesh.conll -v
        
    Kappa score:
        Kappa coefficient for ['B-GENERAL', 'I-GENERAL'] =  0.869
        Kappa coefficient for ['B-PROFANITY', 'I-PROFANITY'] =  0.873
        Kappa coefficient for ['B-VIOLENCE', 'I-VIOLENCE'] =  0.923
        Kappa coefficient for ['B-FEEDBACK', 'I-FEEDBACK'] =  0.400
        Kappa coefficient for ['B-PER', 'I-PER'] =  1.000   
        Kappa coefficient for ['B-MISC', 'I-MISC'] =  1.000
        Kappa coefficient for ['B-GENERAL', 'I-GENERAL', 'B-PROFANITY', 'I-PROFANITY', 'B-VIOLENCE', 'I-VIOLENCE', 'B-FEEDBACK', 'I-FEEDBACK', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'] =  0.782

'''

import os
import argparse
import pandas as pd
import numpy as np
import csv
import shutil

from sklearn.metrics import cohen_kappa_score as kappa

try:
    import conlleval_perl as e
    import utilities as utilities
except ImportError:
    import utility.conlleval_perl as e
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
                    # To know which line is defunct in file
#                     print("Defunct line on first file", i1+1)
#                     print("Defunct line on second file", i2+1)
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


def getTokens(input_1, input_2, output_file):
    """
        Gets labels from CoNLL files from two annotators.
        
        Parameters
        ----------
        input_1 : input file from first annotator
        input_1 : input file from second annotator
        verbose : boolean
            Whether to print out which files are being processed
            
        Returns
        -------
        a1_tokens : list
            List of tokens from first annotator
        a2_tokens : list
            List of tokens from second annotator
        
    """
    a1_tokens = []
    a2_tokens = []
    with open(input_1,'r', encoding='utf-8') as first_file, open(input_2,'r', encoding='utf-8') as second_file, open(output_file,'w', encoding='utf-8') as out_f:
        for (i1,row1),(i2,row2) in zip(enumerate(first_file), enumerate(second_file)):
            # To know which line is defunct in file
#             print("Defunct line on first file", i1+1)
#             print("Defunct line on second file", i2+1)
            if len(row1.split()) > 3 and len(row2.split()) > 3:
                row1 = row1.split()
                row2 = row2.split()
                token = row1[3]
                a1_tokens.append(row1[0])
                a2_tokens.append(row2[0])
                out_f.write(token+'\t'+row1[0]+'\t'+row2[0]+'\n')
            else:
                out_f.write('\n')
    return a1_tokens, a2_tokens

        
def main(**args):
    input_dir_1 = args["input_dir_1"]
    input_dir_2 = args["input_dir_2"]
    input_type = args["input_type"]
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
    if input_type == 'dir':
        print("***************Merging Files***************") 
        a1_tokens, a2_tokens = merger(input_dir_1, input_dir_2, output_file, verbose)
    else:
        a1_tokens, a2_tokens = getTokens(input_dir_1, input_dir_2, output_file)
    
    # Start evaluating merged conll file
    logger = utilities.get_logger(log_file)
    print("***************F1 Evaluation Metric***************")
    e.evaluate_conll_file(logger=logger, fileName=output_file, raw=True, delimiter=None, oTag='O', latex=False)
    
    print("***************Kappa Evaluation Metric***************")
    #     labels=['B-GENERAL','I-GENERAL', 'B-PROFANITY','I-PROFANITY', 'B-VIOLENCE','I-VIOLENCE', 'B-FEEDBACK','I-FEEDBACK', 'B-OUTOFSCOPE','I-OUTOFSCOPE', 'B-PER','I-PER', 'B-ORG','I-ORG', 'B-LOC','I-LOC', 'B-MISC','I-MISC']
    
    highlevel_labels = ['GENERAL', 'PROFANITY', 'VIOLENCE', 'FEEDBACK', 'PER', 'ORG', 'LOC', 'MISC']
    
    label = []
    for each in highlevel_labels:
        per_label = []
        B_label = 'B-'+each
        I_label = 'I-'+each
        label.append(B_label)
        label.append(I_label)
        per_label.append(B_label)
        per_label.append(I_label)
        print_kappa(a1_tokens, a2_tokens, per_label, logger)
    print_kappa(a1_tokens, a2_tokens, label, logger)
    

def print_kappa(a1_tokens, a2_tokens, label, logger):
    logger.info("Kappa coefficient for {0} = {1:6.3f}".format(label, kappa(a1_tokens, a2_tokens, labels=label)))

if __name__=="__main__":
    parser = argparse.ArgumentParser("CoNLL file Merge Argument Parser")
    parser.add_argument("-i", "--input_dir_1", default="./data/agreement/oyesh", metavar="PATH", help="Input first dir path")
    parser.add_argument("-j", "--input_dir_2", default="./data/agreement/sandesh", metavar="PATH", help="Input second dir path")
    parser.add_argument("-o", "--output_file", default="./data/agreement/merged.conll", metavar="FILE", help="Output File Name")
    parser.add_argument("-t", "--input_type", type=str, choices=['dir','file'], default='dir', metavar="FILE", help="Input type: dir or file [default: file]")
    parser.add_argument("-l", "--log_file", default="./data/logs/annotation.log", metavar="FILE", help="Log File Name")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, help="Print description")

    args = vars(parser.parse_args())

    main(**args)
