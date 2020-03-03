'''
    Writes result into the file
    Author: Oyesh Mann Singh
'''

import os
import logging
import numpy as np

import torchtext
from torchtext import data
from torchtext import vocab

import torch
import torch.nn as nn

from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')

import utility.conlleval_perl as e

from sklearn.metrics import precision_recall_fscore_support

class Evaluator():
    def __init__(self, config, logger, model, dataloader, model_name):
        self.config = config
        self.logger = logger
        self.model = model
        self.model_name = model_name
        self.dataloader = dataloader
        
        self.train_dl, self.val_dl, self.test_dl = dataloader.load_data(batch_size=1, shuffle=False)
        self.results_dir = config.results_dir
        
        ts_file = self.model_name+'_test.txt'
        self.test_file = os.path.join(self.results_dir, ts_file)
        
        self.average = config.average
        
        
    def numpy_to_sent(self, tensor):
        '''
            Returns the corresponding TEXT of given Predictions
            Returns chunks of string
        '''    
        return ' '.join([self.dataloader.txt_field.vocab.itos[i] for i in tensor.cpu().data.numpy()[0]]).split()


    def pred_to_tag(self, predictions):
        '''
            Returns the corresponding LABEL of given Predictions
            Returns chunks of string
        '''
        if self.config.train_type == 3:        
            return ' '.join([self.dataloader.ss_field.vocab.itos[i] for i in predictions])       
        else:
            return ' '.join([self.dataloader.ac_field.vocab.itos[i] for i in predictions])       
        
        
    def write_results(self):
        """
        Writes the result into the file
        """        
        with open(self.test_file, 'w', encoding='utf-8') as rtst:
            self.logger.info('Writing in file: {0}'.format(self.test_file))
            tt = tqdm(iter(self.test_dl), leave=False)
            for ((y, ac, at, X), v) in tt:
#                 print(vars(self.test_dl.dataset.examples[0]))
#                 print(X.shape)
#                 print(aspect.shape)
                if self.config.train_type == 3:
                    pred = self.model(X, at, ac)
                else:
                    pred = self.model(X, at, None)
                sent = self.numpy_to_sent(X)
                sent = ' '.join(sent)
                
                pred_idx = pred.argmax(dim = 1)

                y = y.squeeze(1)
                y_true_val = y.cpu().data.numpy()
                true_tag = self.pred_to_tag(y_true_val)

                y_pred_val = pred_idx.cpu().data.numpy()
                pred_tag = self.pred_to_tag(y_pred_val)

                rtst.write(sent+'\t'+true_tag+'\t'+pred_tag+'\n')
                
                rtst.write('\n')
        rtst.close()
        
        
    def prec_rec_f1(self, gold_list, pred_list):
        """
        Calculates the precision, recall, f1 score
        :param gold_list: gold labels        
        :param pred_list: predicted labels
        :return: precision, recall, f1 score
        """
        prec, rec, f1, _ = precision_recall_fscore_support(gold_list, pred_list, average=self.average)
        return prec, rec, f1
