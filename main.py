#!/usr/bin/env python3
'''
    Main file
    Author: Oyesh Mann Singh
    
    How to run:
    For training:
        python main.py -t 2 -k 1 -d cpu
        
    For evaluation:
        python main.py -k 1 -e
'''

import os
import argparse
import shutil
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import numpy as np
from utility.dataloader import Dataloader
import utility.utilities as utilities
import utility.splitter as splitter

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
    parser.add_argument("-t", "--train_type", type=int, choices=[1,2], default=1, 
                        help="1: Text-> Label, 2: Text+AspectTerm -> AspectCategory")
    parser.add_argument("-m", "--model", type=str, choices=['lstm','cnn'], default='lstm', 
                        help="LSTM or CNN model [default: LSTM]")    
    parser.add_argument("-k", "--kfold", dest="kfold", type=int, 
                        default=1, metavar="INT", help="K-fold cross validation [default:1]")

    
    args = parser.parse_args()
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    os.mkdir(args.log_dir)   

    # Init Logger
    log_file = os.path.join(args.log_dir, 'complete.log')
    data_log = os.path.join(args.log_dir, 'data_log.log')
    logger = utilities.get_logger(log_file)
    
    config = Configuration(config_file=args.config_file, logger=logger)
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
    
    logger.info("***************************************")
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
    logger.info("***************************************") 

    return config, logger



def main():
    """
        Main File
    """
    # Parse argument
    config, logger = parse_args()
    
    # Splits the given dataset into k-fold
    if config.kfold > 0 and not config.eval:
        logger.info("Splitting dataset into {0}-fold".format(config.kfold))
        splitter.main(input_file = config.data_file, 
                      output_dir = config.root_path, 
                      verbose    = config.verbose, 
                      kfold      = config.kfold,
                      csv        = config.csv,
                      log_file   = config.data_log)
    
    tot_acc = 0
    tot_prec = 0
    tot_rec = 0
    tot_f1 = 0
    
    # Training for each fold
    for i in range(0, config.kfold):
        # To match the output filenames
        k = str(i+1)
        
        if not config.eval:
            logger.info("Starting training on {0}th-fold".format(k))
        
        # Load data iterator
        dataloader = Dataloader(config, k)
    
        # Debugging purpose. DO NOT DELETE
#         train_iter, val_iter, test_iter = dataloader.load_data(batch_size=1)
#         sample = next(iter(train_iter))
#         print(sample.TEXT)
#         print(sample.TERM)
#         print(sample.LABEL)
        
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
            model.fit()

        # Test
        logger.info("**************Testing Started !!!**************\n")
        model.load_checkpoint()
        acc, prec, rec, f1 = model.predict()
        logger.info("Accuracy: %6.3f Precision: %6.3f Recall: %6.3f FB1: %6.3f "% (acc, prec, rec, f1))
        
        # Calculate the metrics
        tot_acc += acc
        tot_prec += prec
        tot_rec += rec
        tot_f1 += f1
        
    logger.info("Final Accuracy: %6.3f Final Precision: %6.3f Final Recall: %6.3f Final FB1: %6.3f "% (tot_acc/config.kfold, tot_prec/config.kfold, tot_rec/config.kfold, tot_f1/config.kfold))
        


if __name__=="__main__":
    main()
