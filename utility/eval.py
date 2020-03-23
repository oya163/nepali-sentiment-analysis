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

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

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

    
    def numpy_to_at(self, tensor):
        '''
            Returns the corresponding ASPECT TERM of given Predictions
            Returns chunks of string
        '''    
        return ' '.join([self.dataloader.at_field.vocab.itos[i] for i in tensor.cpu().data.numpy()[0]]).split()
    
    
    def numpy_to_ac(self, tensor):
        '''
            Returns the corresponding ASPECT TERM of given Predictions
            Returns chunks of string
        '''    
        return ' '.join([self.dataloader.ac_field.vocab.itos[i] for i in tensor])       
    
    
    def pred_to_tag(self, predictions):
        '''
            Returns the corresponding LABEL of given Predictions
            Returns chunks of string
        '''
        if self.config.train_type == 3 or self.config.train_type == 4:        
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
#                 print(at.shape)
#                 print(ac.shape)

                pred = self.model(X, at, ac)
                
                sent = self.numpy_to_sent(X)
                sent = ' '.join(sent)
                
                aspect = self.numpy_to_at(at)
                aspect = ' '.join(aspect)  
                
                aspect_cat = self.numpy_to_ac(ac)          
                
                pred_idx = pred.argmax(dim = 1)

                y = y.squeeze(1)
                y_true_val = y.cpu().data.numpy()
                true_tag = self.pred_to_tag(y_true_val)

                y_pred_val = pred_idx.cpu().data.numpy()
                pred_tag = self.pred_to_tag(y_pred_val)

                rtst.write(sent+'\t'+aspect+'\t'+aspect_cat+'\t'+true_tag+'\t'+pred_tag+'\n')
                
                rtst.write('\n')
        rtst.close()   

        
    def infer(self, sent, aspect_term, aspect_cat):
        """
        Prints the result
        """        
        # Tokenize the sentence and aspect terms
        sent_tok = self.dataloader.tokenizer(sent)
        at_tok = self.dataloader.tokenizer(aspect_term)
        
        # Get index from vocab
        X = [self.dataloader.txt_field.vocab.stoi[t] for t in sent_tok]
        at = [self.dataloader.at_field.vocab.stoi[t] for t in at_tok]
        ac = [self.dataloader.ac_field.vocab.stoi[aspect_cat]]
        
        # Convert into torch and reshape into [batch, sent_length]
        X = torch.LongTensor(X).to(self.config.device)
        X = X.unsqueeze(0)
        
        at = torch.LongTensor(at).to(self.config.device) 
        at = at.unsqueeze(0)
        
        ac = torch.LongTensor(ac).to(self.config.device) 
        ac = ac.unsqueeze(0)        

        # Get predictions
        pred = self.model(X, at, ac)

        pred_idx = pred.argmax(dim = 1)

        y_pred_val = pred_idx.cpu().data.numpy()
        pred_tag = self.pred_to_tag(y_pred_val)
        return pred_tag
        
        
    def prec_rec_f1(self, gold_list, pred_list):
        """
        Calculates the precision, recall, f1 score
        :param gold_list: gold labels        
        :param pred_list: predicted labels
        :return: precision, recall, f1 score
        """
        prec, rec, f1, _ = precision_recall_fscore_support(gold_list, pred_list, average=self.average)
        gold_list = np.array(gold_list)
        pred_list = np.array(pred_list)
        
        n_values = np.max(gold_list) + 1
        
        # create one hot encoding for auc calculation
        gold_list = np.eye(n_values)[gold_list]
        pred_list = np.eye(n_values)[pred_list]
        auc = roc_auc_score(gold_list, pred_list, average=self.average, multi_class=self.config.auc_multiclass)
        return prec, rec, f1, auc
