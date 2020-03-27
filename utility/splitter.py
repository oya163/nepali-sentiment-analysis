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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

try:
    import utilities as utilities
except ImportError:
    import utility.utilities as utilities

MAX_SEQ_LENGTH = 200
MIN_SEQ_LENGTH = 5

#forbidden_list = ['B-DATE','I-DATE', 'B-EVENT','I-EVENT', 'B-NUM','I-NUM', 'B-SARCASM','I-SARCASM', 'B-OUTOFSCOPE','I-OUTOFSCOPE']

#forbidden_list = ['B-FEEDBACK','I-FEEDBACK', 'B-DATE','I-DATE', 'B-EVENT','I-EVENT', 'B-NUM','I-NUM', 'B-SARCASM','I-SARCASM', 'B-OUTOFSCOPE','I-OUTOFSCOPE', 'B-GENERAL','I-GENERAL', 'B-PROFANITY','I-PROFANITY', 'B-VIOLENCE','I-VIOLENCE']

forbidden_list = ['B-DATE','I-DATE', 'B-EVENT','I-EVENT', 'B-NUM','I-NUM', 'B-SARCASM','I-SARCASM', 'B-OUTOFSCOPE','I-OUTOFSCOPE', 'B-PER','I-PER', 'B-ORG','I-ORG', 'B-LOC','I-LOC', 'B-MISC','I-MISC']


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
        min_counter=0
        sent_counter=0
        line_num=0
        j=0
        for i,row in enumerate(in_file):
            #To know which line is defunct in file
            #print(i+1)
            row = row.strip().split()
            
            # Assuming input file has four columns
            # token, start_position, end_position, entity_type
            if len(row)==4:
                sentence.append(row[0])
                tag.append(row[-1])
            else:
                line_num+=1
                if len(sentence) > max_length:
                    max_length = len(sentence)
                    max_sentence=sentence
                    j=line_num
                
                if len(sentence) < MAX_SEQ_LENGTH and len(sentence) > MIN_SEQ_LENGTH:
                    txt_f.write(' '.join(sentence)+'\n')
                    tag_f.write(' '.join(tag)+'\n')
                    sent_counter+=1
                else:
                    if len(sentence) > MAX_SEQ_LENGTH:
                        max_counter+=1
                        if verbose:
                            logger.info("Length of longer sentence = {}".format(len(sentence)))
                    else:
                        min_counter+=1
                        if verbose:
                            logger.info("Length of shorter sentence = {}".format(len(sentence)))

                sentence = []
                tag = []                   
            

        if verbose:
            logger.info("Max sentence length limit = {}".format(MAX_SEQ_LENGTH))
            logger.info("Min sentence length limit = {}".format(MIN_SEQ_LENGTH))
            logger.info("Longest sentence length = {}".format(max_length))
            logger.info("Longest sentence at line number = {}".format(j))
            logger.info("Longest sentence counter = {}".format(max_counter))
            logger.info("Shortest sentence counter = {}".format(min_counter))
            logger.info("% of sentence removed = {}%".format(max_counter+min_counter/line_num * 100))
            logger.info("Total number of sentence before removal= {}".format(line_num))
            logger.info("Total number of sentence after removal= {}".format(sent_counter))
            
        in_file.close()
        txt_f.close()
        tag_f.close()
        logger.info("Text and Tag files are stored in {}".format(output_dir))
        logger.info("******************************************************")
        return sent_file, tag_file


'''
    Function to write dataframe into files
'''
def write_df(df, fname, logger):
    invalid_counter = 0
    with open(fname, 'w', encoding='utf-8') as f:
        for i, r in df.iterrows():
            # Splits the TEXT and TAG into chunks
            text = r['TEXT'].split()
            tag = r['TAG'].split()
            tag = ['O' if x in forbidden_list else x for x in tag]
            
            # Remove specific lines having these categories
            # if not set(tag).intersection(set(['B-SARCASM','I-SARCASM', 'B-OUTOFSCOPE','I-OUTOFSCOPE'])):
            # Remove if it contains only 'O'
            if list(set(tag)) != ['O']:
                for t1, t2 in zip(text, tag):
                    f.write(t1+'\t'+t2+'\n')
            else:
                invalid_counter+=1
            f.write('\n')
        logger.info('Number of sentences containing only \'O\': {}'.format(invalid_counter))
        logger.info('Created: {}'.format(fname))
        f.close()

    return invalid_counter

        
'''
    Partitions the given data into chunks
    Create train/test file accordingly
'''
def split_train_test(source_path, save_path, logger):
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

    df = df_txt.join(df_tag).sample(frac=1).reset_index(drop=True)
    
    # To split into train and intermediate 80/20
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask]
    intermediate_df = df[~mask]
    
    # To split intermediate into 10/10 into test and dev
    val_mask = np.random.rand(len(intermediate_df)) < 0.5
    test_df = intermediate_df[val_mask]
    val_df = intermediate_df[~val_mask]

    # Write those train/test dataframes into files
    invalid_train_count = write_df(train_df, train_fname, logger)
    invalid_test_count = write_df(test_df, test_fname, logger)
    invalid_val_count = write_df(val_df, val_fname, logger)
    
    # Print stat
    logger.info("Length of train dataset: {}".format(len(train_df) - invalid_train_count))
    logger.info("Length of test dataset: {}".format(len(test_df) - invalid_test_count))
    logger.info("Length of val dataset: {}".format(len(val_df) - invalid_val_count))

'''
    Partitions the given data into chunks
    Create train/test file accordingly
    
    ***Obsolete yet for reference***
'''
def split_train_test_csv(source_path, save_path, logger):
    
    logger.info("Saving path: {}".format(save_path))
        
    train_fname = os.path.join(save_path,'train.txt')
    test_fname = os.path.join(save_path, 'test.txt')
    val_fname = os.path.join(save_path, 'val.txt')
    
    df_txt = pd.read_csv(source_path, delimiter=',', encoding='utf-8', 
                         skip_blank_lines=True, header=['ss', 'ac', 'at', 'text'], 
                         quoting=csv.QUOTE_MINIMAL, names=['TEXT'])

    df = df_txt.sample(frac=1).reset_index(drop=True)
    
    # To split into train and intermediate 80/20
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask]
    intermediate_df = df[~mask]
    
    # To split intermediate into 10/10 into test and dev
    val_mask = np.random.rand(len(intermediate_df)) < 0.5
    test_df = intermediate_df[val_mask]
    val_df = intermediate_df[~val_mask]
   
    
    train_df.to_csv(train_fname, header=False, index=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar=" ", encoding='utf-8')
    test_df.to_csv(test_fname, header=False, index=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar=" ", encoding='utf-8')
    val_df.to_csv(val_fname, header=False, index=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar=" ", encoding='utf-8')

    # Print stat
    logger.info("Length of train dataset: {}".format(len(train_df)))
    logger.info("Length of test dataset: {}".format(len(test_df)))
    logger.info("Length of val dataset: {}".format(len(val_df)))



def write_csv(df, fname):
    df.to_csv(fname, header=False, index=False, 
                    quoting=csv.QUOTE_MINIMAL,  
                    escapechar=" ", 
                    encoding='utf-8')
    
    
'''
    Partitions the given data using GroupShuffleSplit
    
    This function will split train/test/val for each
    aspect category equally
    
    Split 80/10/10 for all the category
    
    ** Not based on the whole document
'''
def split_csv(source_path, save_path, logger):

    logger.info("Saving path: {}".format(save_path))
        
    train_fname = os.path.join(save_path,'train.txt')
    test_fname = os.path.join(save_path, 'test.txt')
    val_fname = os.path.join(save_path, 'val.txt')
    
    df_txt = pd.read_csv(source_path, delimiter=',', 
                         encoding='utf-8', 
                         skip_blank_lines=True, 
                         header=None, 
                         names=['ss', 'ac', 'at', 'text'])

    # Split the df based on sentiment strength
    # into positive and negative
    gss = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 163).split(df_txt, groups=df_txt['ss'])

    # Get positive and negative dataframe
    for positive_df, negative_df in gss:
        
        # Get data based on the index
        negative = df_txt.iloc[negative_df]
        positive = df_txt.iloc[positive_df]

        # Split 80/10/10 -> train, test, val
        # based on sentiment strength
        train_neg, test_val_neg = train_test_split(negative, test_size=0.2)
        train_pos, test_val_pos = train_test_split(positive, test_size=0.2)
        test_neg, val_neg = train_test_split(test_val_neg, test_size=0.5)
        test_pos, val_pos = train_test_split(test_val_pos, test_size=0.5)

        # Concat negative and positive dataframe and shuffle
        train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)
        test_df = pd.concat([test_pos, test_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)
        val_df = pd.concat([val_pos, val_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)

        # Write into csv file
        write_csv(train_df, train_fname)
        write_csv(test_df, test_fname)
        write_csv(val_df, val_fname)

    # Print stat
    logger.info("******************************************************")
    logger.info("Length of train dataset: {}".format(len(train_df)))
    logger.info("Length of test dataset: {}".format(len(test_df)))
    logger.info("Length of val dataset: {}".format(len(val_df)))
    
    logger.info("Train dataset groupby aspect category: \n{}".format(train_df.groupby('ac').count()))
    logger.info("Test dataset groupby aspect category: \n{}".format(test_df.groupby('ac').count()))
    logger.info("Val dataset groupby aspect category: \n{}".format(val_df.groupby('ac').count()))
    logger.info("******************************************************")

    
def split(input_file, save_path, verbose, logger):
    sent_file, tag_file = text_tag_convert(input_file, logger, verbose)
    
    source_path = os.path.dirname(sent_file)
    logger.info("Source path: {}".format(source_path))
    split_train_test(source_path, save_path, logger)

        
def main(**args):
    input_file = args["input_file"]
    save_path = args["output_dir"]
    verbose = args["verbose"]
    kfold = args["kfold"]
    csv = args["csv"]
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
        if not csv:
            split(input_file, final_path, verbose, logger)
        else:
            split_csv(input_file, final_path, logger)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser("Dataset Splitter Argument Parser")
    parser.add_argument("-i", "--input_file", default="./data/dataset/total.conll", metavar="PATH", help="Input file path")
    parser.add_argument("-o", "--output_dir", default="../torchnlp/data/nepsa/", metavar="PATH", help="Output Directory")
    parser.add_argument("-c", "--csv", action='store_true', default=False, help="CSV file splitter")
    parser.add_argument("-k", "--kfold", dest='kfold', type=int, default=1, metavar="INT", help="K-fold")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, help="Print description")
    parser.add_argument("-l", "--log_file", dest="log_file", type=str, metavar="PATH", default="./logs/data_log.log",help="Log file")

    args = vars(parser.parse_args())

    main(**args)
