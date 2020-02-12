#!/usr/bin/env python3
'''
    Splits dataset into train/test/val
    Author: Oyesh Mann Singh
    Date: 10/16/2019
'''

import os
import argparse
import pandas as pd
import numpy as np
import csv
import shutil

try:
    import utilities as utilities
except ImportError:
    import utils.utilities as utilities

MAX_SEQ_LENGTH = 200

def text_tag_convert_with_pos(input_file, logger, verbose=False):
    dir_name = os.path.dirname(input_file)
    
    output_dir = os.path.join(dir_name, 'text_tag_only')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    sent_file = os.path.join(output_dir, 'text_only.txt')
    tag_file = os.path.join(output_dir, 'tag_only.txt')
    pos_file = os.path.join(output_dir, 'pos_only.txt')
                         
    
    with open(input_file,'r', encoding='utf-8') as in_file, open(sent_file,'w', encoding='utf-8') as txt_f, open(tag_file,'w', encoding='utf-8') as tag_f, open(pos_file,'w', encoding='utf-8') as pos_f:
        sentence = []
        tag = []
        pos = []
        max_length=0
        max_sentence=''
        max_counter=0
        line_num=0
        j=0
        for i,row in enumerate(in_file):
            #To know which line is defunct in file
            #print(i+1)
            row = row.strip().split("\t")

            if len(row) > 2:
                sentence.append(row[0])
                pos.append(row[1])
                tag.append(row[-1])
            else:
                line_num+=1
                if len(sentence) > max_length:
                    max_length = len(sentence)
                    max_sentence=sentence
                    j=line_num
                
                if len(sentence) < MAX_SEQ_LENGTH:
                    txt_f.write(' '.join(sentence)+'\n')
                    tag_f.write(' '.join(tag)+'\n')
                    pos_f.write(' '.join(pos)+'\n')
                else:
                    max_counter+=1
                    logger.info("Length of longer sentence = {}".format(len(sentence)))
                sentence = []
                tag = []                 
                pos = []


        if verbose:
            logger.info("Max sentence length limit = {}".format(MAX_SEQ_LENGTH))
            logger.info("Longest sentence length = {}".format(max_length))
            logger.info("Longest sentence at line number = {}".format(j))
            logger.info("Sentence having higher counter = {}".format(max_counter))
            logger.info("Total number of sentence = {}".format(line_num))
            logger.info("% of sentence removed = {}%".format(max_counter/line_num * 100))
            
        in_file.close()
        txt_f.close()
        tag_f.close()
        pos_f.close()
        logger.info("Text, POS and Tag files are stored in {}".format(output_dir))
        logger.info("******************************************************")
        return sent_file, pos_file, tag_file


def text_tag_convert(input_file, logger, verbose=False):
    dir_name = os.path.dirname(input_file)
    
    output_dir = os.path.join(dir_name, 'text_tag_only')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    sent_file = os.path.join(output_dir, 'text_only.txt')
    tag_file = os.path.join(output_dir, 'tag_only.txt')
                         
    
    with open(input_file,'r', encoding='utf-8') as in_file, open(sent_file,'w', encoding='utf-8') as txt_f, open(tag_file,'w', encoding='utf-8') as tag_f:
        sentence = []
        tag = []
        max_length=0
        max_sentence=''
        max_counter=0
        line_num=0
        j=0
        for i,row in enumerate(in_file):
            #To know which line is defunct in file
            #print(i+1)
            row = row.strip().split("\t")

            if len(row)>1:
                sentence.append(row[0])
                tag.append(row[-1])
            else:
                line_num+=1
                if len(sentence) > max_length:
                    max_length = len(sentence)
                    max_sentence=sentence
                    j=line_num
                
                if len(sentence) < MAX_SEQ_LENGTH:
                    txt_f.write(' '.join(sentence)+'\n')
                    tag_f.write(' '.join(tag)+'\n')
                else:
                    max_counter+=1
                    logger.info("Length of longer sentence = {}".format(len(sentence)))
                sentence = []
                tag = []                 


        if verbose:
            logger.info("Max sentence length limit = {}".format(MAX_SEQ_LENGTH))
            logger.info("Longest sentence length = {}".format(max_length))
            logger.info("Longest sentence at line number = {}".format(j))
            logger.info("Sentence having higher counter = {}".format(max_counter))
            logger.info("Total number of sentence = {}".format(line_num))
            logger.info("% of sentence removed = {}%".format(max_counter/line_num * 100))
            
        in_file.close()
        txt_f.close()
        tag_f.close()
        logger.info("Text and Tag files are stored in {}".format(output_dir))
        logger.info("******************************************************")
        return sent_file, tag_file


'''
    Function to write dataframe into files
'''
def write_df(df, fname, logger, use_pos=False):
    with open(fname, 'w', encoding='utf-8') as f:
        for i, r in df.iterrows():
            # Splits the TEXT and TAG into chunks
            text = r['TEXT'].split()
            tag = r['TAG'].split()
            if use_pos:
                pos = r['POS'].split()
                for t1, t2, t3 in zip(text, pos, tag):
                    f.write(t1+'\t'+t2+'\t'+t3+'\n')                
            else:
                for t1, t2 in zip(text, tag):
                    f.write(t1+'\t'+t2+'\n')
            f.write('\n')
        logger.info('Created: {}'.format(fname))
        f.close()

        
'''
    Partitions the given data into chunks
    Create train/test file accordingly
'''
def split_train_test(source_path, save_path, logger, pos):
    sent_file = os.path.join(source_path, 'text_only.txt')
    tag_file = os.path.join(source_path, 'tag_only.txt')
    
    logger.info("Saving path: {}".format(save_path))
    
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
        
    train_fname = os.path.join(save_path,'train.txt')
    test_fname = os.path.join(save_path, 'test.txt')
    val_fname = os.path.join(save_path, 'val.txt')
    
    df_txt = pd.read_csv(sent_file, delimiter='\n', encoding='utf-8', 
                         skip_blank_lines=True, header=None, 
                         quoting=csv.QUOTE_NONE, names=['TEXT'])
    
    df_tag = pd.read_csv(tag_file, delimiter='\n', encoding='utf-8', 
                         skip_blank_lines=True, header=None, 
                         quoting=csv.QUOTE_NONE, names=['TAG'])
    if pos:
        pos_file = os.path.join(source_path, 'pos_only.txt')
        df_pos = pd.read_csv(pos_file, delimiter='\n', encoding='utf-8', 
                             skip_blank_lines=True, header=None, 
                             quoting=csv.QUOTE_NONE, names=['POS'])
        df = [df_txt, df_pos, df_tag]
        df = df[0].join(df[1:]).sample(frac=1).reset_index(drop=True)
    else:
        df = df_txt.join(df_tag).sample(frac=1).reset_index(drop=True)
    
    # To split into train and test 70/30
    mask = np.random.rand(len(df)) < 0.7
    train_df = df[mask]
    intermediate_df = df[~mask]
    
    # To split intermediat into 50/50
    val_mask = np.random.rand(len(intermediate_df)) < 0.5
    test_df = intermediate_df[val_mask]
    val_df = intermediate_df[~val_mask]

    # Write those train/test dataframes into files
    write_df(train_df, train_fname, logger, pos)
    write_df(test_df, test_fname, logger, pos)
    write_df(val_df, val_fname, logger, pos)
    
    # Print stat
    logger.info("Length of train dataset: {}".format(len(train_df)))
    logger.info("Length of test dataset: {}".format(len(test_df)))
    logger.info("Length of val dataset: {}".format(len(val_df)))


def split(input_file, save_path, verbose, logger, pos):
    if pos:
        sent_file, pos_file, tag_file = text_tag_convert_with_pos(input_file, logger, verbose)
    else:
        sent_file, tag_file = text_tag_convert(input_file, logger, verbose)
    
    source_path = os.path.dirname(sent_file)
    logger.info("Source path: {}".format(source_path))
    split_train_test(source_path, save_path, logger, pos)

        
def main(**args):
    input_file = args["input_file"]
    save_path = args["output_dir"]
    verbose = args["verbose"]
    kfold = args["kfold"]
    pos = args["pos"]
    log_file = args["log_file"]
    
    logger = utilities.get_logger(log_file)
    
    # Clean up output directory
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.mkdir(save_path)
    
    # Start splitting dataset
    # into respective directory
    for i in range(0, kfold):
        final_path = os.path.join(save_path, str(i+1))
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        split(input_file, final_path, verbose, logger, pos)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser("Dataset Splitter Argument Parser")
    parser.add_argument("-i", "--input_file", default="./data/umbc/stemmed/total.txt", metavar="PATH", help="Input file path")
    parser.add_argument("-o", "--output_dir", default="./data/umbc/stemmed/kfold", metavar="PATH", help="Output Directory")
    parser.add_argument("-p", "--pos", action='store_true', default=False, help="Use POS")
    parser.add_argument("-k", "--kfold", dest='kfold', type=int, default=1, metavar="INT", help="K-fold")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, help="Print description")
    parser.add_argument("-l", "--log_file", dest="log_file", type=str, metavar="PATH", default="./logs/data_log.txt",help="Log file")

    args = vars(parser.parse_args())

    main(**args)
