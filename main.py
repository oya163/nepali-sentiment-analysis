#!/usr/bin/env python3
'''
    Main file
    Author: Oyesh Mann Singh
    
    How to run:
    For training:
        python main.py -t 3 -k 1 -d cpu
        
    For evaluation:
        python main.py -k 1 -e
'''

import os
import sys
import argparse
import shutil
import logging
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import numpy as np
from utility.dataloader import Dataloader
import utility.utilities as utilities
import utility.splitter as splitter
from utility.eval import Evaluator

from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config.config import Configuration
from models.models import LSTM, CNN
from train import Trainer

def parse_args():
    """
        Argument Parser
    """
    parser = argparse.ArgumentParser(description="NepSA Main Parser")
    
    parser.add_argument("-c", "--config", dest="config_file", type=str, metavar="PATH", 
                        default="./config/config.ini",help="Configuration file path")
    parser.add_argument("-r", "--root_path", dest="root_path", type=str, metavar="PATH", 
                        default=None,help="Data root file path")    
    parser.add_argument("-l", "--log_dir", dest="log_dir", type=str, 
                        metavar="PATH", default="./logs",help="Log file path")    
    parser.add_argument("-d", "--device", dest="device", type=str, 
                        default="cuda:0", help="device[‘cpu’,‘cuda:0’,‘cuda:1’,..]")
    parser.add_argument("-v", "--verbose", action='store_true', 
                        default=False, help="Print data description")
    parser.add_argument("-s", "--csv", action='store_true', 
                        default=True, help="CSV file splitter")
    parser.add_argument("-e", "--eval", action='store_true', 
                        default=False, help="For evaluation purpose only")
    parser.add_argument("-i", "--infer", action='store_true',
                        default=False, help="For inference purpose only")    
    parser.add_argument("-t", "--train_type", type=int, choices=[1,2,3,4], default=3, 
                        help="""1: Text-> AspectCategory, 
                                2: Text+AspectTerm -> AspectCategory,
                                3: Text+AspectTerm+AspectCategory -> SS,
                                4: Text -> SS""") 
    parser.add_argument("-m", "--model", type=str, choices=['lstm','cnn'], default='lstm', 
                        help="LSTM or CNN model [default: LSTM]")
    parser.add_argument("-k", "--kfold", dest="kfold", type=int, 
                        default=1, metavar="INT", help="K-fold cross validation [default:1]")
    parser.add_argument("-n", "--model_name", dest="model_name", type=str, 
                        default='', metavar="PATH", help="Model file name")
    parser.add_argument("--txt", dest="txt", type=str, 
                        default="रबि लामिछाने नेपालि जन्ता को हिरो हुन", help="Input text (For inference purpose only)")    
    parser.add_argument("--at", dest="at", type=str, 
                        default="हिरो हुन", help="Input aspect term (For inference purpose only)") 
    parser.add_argument("--ac", dest="ac", type=str, 
                        default='GENERAL', help="Input aspect category (For inference purpose only)")    
    
    args = parser.parse_args()
    
    # If log dir does not exist, create it
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)   

    # Init Logger
    log_suffix = '_'+args.model+'_'+str(args.train_type)+'.log'
    log_file = os.path.join(args.log_dir, 'complete'+log_suffix)
    data_log = os.path.join(args.log_dir, 'datalog'+log_suffix)
    
    # If log files exist, remove them
    if os.path.exists(log_file):
        os.remove(log_file) 
        
    if os.path.exists(data_log):
        os.remove(data_log) 
        
    # Logger
    logger = utilities.get_logger(log_file)
    
    # Configuration
    config = Configuration(config_file=args.config_file, logger=logger, args=args)
    config.device = args.device
    config.verbose = args.verbose
    config.eval = args.eval
    config.kfold = args.kfold
    config.log_dir = args.log_dir
    config.log_file = log_file
    config.data_log = data_log
    config.csv = args.csv
    config.train_type = args.train_type
    config.model = args.model
    model_filename = os.path.basename(config.data_file).split('.')[0]+'_'+config.model+'_'+str(config.train_type)
    config.model_name = args.model_name if args.model_name else model_filename
    config.root_path = args.root_path if args.root_path else os.path.join(config.data_path, config.model_name)
    config.infer = args.infer
    config.txt = args.txt
    config.at = args.at
    config.ac = args.ac
    
    logger.info("*******************************ARGS")
    logger.info("Data file : {}".format(config.data_file))
    logger.info("Device : {}".format(config.device))
    logger.info("Verbose : {}".format(config.verbose))
    logger.info("Eval mode : {}".format(config.eval))
    logger.info("K-fold : {}".format(config.kfold))    
    logger.info("Log directory: {}".format(config.log_dir))     
    logger.info("Data log file: {}".format(config.data_log))
    logger.info("Split csv file: {}".format(config.csv))
    logger.info("Training Type: {}".format(config.train_type))    
    logger.info("Model: {}".format(config.model))
    logger.info("Model name: {}".format(config.model_name))
    logger.info("Root path: {}".format(config.root_path))
    logger.info("Inference mode: {}".format(config.infer))
    if config.infer:
        logger.info("Text: {}".format(config.txt))
        logger.info("Aspect Term: {}".format(config.at))
        logger.info("Aspect Category: {}".format(config.ac))        
    logger.info("***************************************")
    
    return config, logger


# Inference section
def infer(config, logger):
    k = str(config.kfold)
    dataloader = Dataloader(config, k)
    
    # Load model
    arch = LSTM(config, dataloader).to(config.device)
    if config.model == 'cnn':
        arch = CNN(config, dataloader).to(config.device)    
        
    # Print network configuration
    logger.info(arch)

    # Trainer
    model = Trainer(config, logger, dataloader, arch, k)
    
    model.load_checkpoint()
    
    logger.info("Inferred results")
    
    pred_tag = model.infer(config.txt, config.at, config.ac)
    
    print(config.txt+'\t'+config.at+'\t'+config.ac+'\t'+pred_tag+'\n')
    

# Train/test section
def train_test(config, logger):

    tot_acc = 0
    tot_prec = 0
    tot_rec = 0
    tot_f1 = 0
    tot_auc = 0
    
    total_start_time = time.time()
    # Training for each fold
    for i in range(0, config.kfold):
        # To match the output filenames
        k = str(i+1)
        
        # Load data iterator
        dataloader = Dataloader(config, k)
        
        # Debugging purpose. DO NOT DELETE
#         train_iter, val_iter, test_iter = dataloader.load_data(batch_size=1)
#         e = Evaluator(config, None, None, dataloader, 'debug')
        
#         for ((y, ac, at, X), v) in train_iter:
#             print("TEXT = ", e.numpy_to_sent(X))
#             print("ASPECT TERM = ", e.numpy_to_at(at))
#             print("ASPECT CATEGORY = ", e.numpy_to_ac(ac))
#             print("SENTIMENT STRENGTH = ", e.pred_to_tag(y))
        
#         sample = next(iter(train_iter))
#         print(sample.TEXT)
#         print("TEXT = ", e.numpy_to_sent(sample.TEXT))
#         print("ASPECT TERM = ", e.numpy_to_sent(sample.TERM))

#         for i,each in enumerate(iter(train_iter)):
#             print("TEXT = ", train_iter.dataset.examples[i].TEXT)
#             print("TERM = ",train_iter.dataset.examples[i].TERM)
#             print("ASPECT = ",train_iter.dataset.examples[i].ASPECT)
#             print("SS = ",train_iter.dataset.examples[i].SS)
        
        #### Just Run this to check the values ####
#         print("TEXT = ", train_iter.dataset.examples[0].TEXT)
#         print("TERM = ",train_iter.dataset.examples[0].TERM)
#         print("ASPECT = ",train_iter.dataset.examples[0].ASPECT)
#         print("SS = ",train_iter.dataset.examples[0].SS)
#         break

        # Load model
        arch = LSTM(config, dataloader).to(config.device)
        if config.model == 'cnn':
            arch = CNN(config, dataloader).to(config.device)

        # Print network configuration
        logger.info(arch)
        
        # Trainer
        model = Trainer(config, logger, dataloader, arch, k)
        
        # Train
        if not config.eval:
            logger.info("**************Training started !!!**************\n")
            logger.info("Starting training on {0}th-fold".format(k))
            model.fit()

        # Test
        logger.info("**************Testing Started !!!**************\n")
        model.load_checkpoint()
        acc, prec, rec, f1, auc = model.predict()
        logger.info("Accuracy: %6.3f Precision: %6.3f Recall: %6.3f FB1: %6.3f AUC: %6.3f"% (acc, prec, rec, f1, auc))
        logger.info("***********************************************\n")
        
        # Calculate the metrics
        tot_acc += acc
        tot_prec += prec
        tot_rec += rec
        tot_f1 += f1
        tot_auc += auc
    
    total_end_time = time.time()
    
    # Display final results
    epoch_mins, epoch_secs = utilities.epoch_time(total_start_time, total_end_time)
    logger.info("Epoch Time: %dm %ds"%(epoch_mins, epoch_secs))
    logger.info("Final_Accuracy;%6.3f;Final_Precision;%6.3f;Final_Recall;%6.3f;Final_FB1;%6.3f;Final_AUC;%6.3f "% (tot_acc/config.kfold, tot_prec/config.kfold, tot_rec/config.kfold, tot_f1/config.kfold, tot_auc/config.kfold))
        

def main():
    """
        Main File
    """
    # Parse argument
    config, logger = parse_args()
    
    if config.infer:
        infer(config, logger)
    else:
        train_test(config, logger)

        
if __name__=="__main__":
    main()
