#!/usr/bin/env python3

'''
    Trainer
    Author: Oyesh Mann Singh
'''

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from utility.dataloader import Dataloader
from utility.eval import Evaluator

from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
torch.manual_seed(163)

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# Decay functions to be used with lr_scheduler
def lr_decay_noam(config):
    return lambda t: (
        10.0 * config.hidden_dim**-0.5 * min(
        (t + 1) * config.learning_rate_warmup_steps**-1.5, (t + 1)**-0.5))

def lr_decay_exp(config):
    return lambda t: config.learning_rate_falloff ** t


# Map names to lr decay functions
lr_decay_map = {
    'noam': lr_decay_noam,
    'exp': lr_decay_exp
}

# Trainer class
class Trainer():
    def __init__(self, config, logger, dataloader, model, k):
        """
            Trainer class
        """        
        self.config = config
        self.logger = logger
        self.dataloader = dataloader
        self.verbose = config.verbose
        
        self.train_dl, self.val_dl, self.test_dl = dataloader.load_data(batch_size=config.batch_size)

        ### DO NOT DELETE
        ### DEBUGGING PURPOSE
#         sample = next(iter(self.train_dl))
#         print(sample.TEXT)
#         print(sample.LABEL)
#         print(sample.POS)
        
        self.train_dlen = len(self.train_dl)
        self.val_dlen = len(self.val_dl)
        self.test_dlen = len(self.test_dl)
        
        self.model = model
        self.epochs = config.epochs
        
        self.loss_fn = nn.CrossEntropyLoss()

        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                         lr=config.learning_rate, 
                         weight_decay=config.weight_decay)
        
        self.lr_scheduler_step = self.lr_scheduler_epoch = None
        
        # Set up learing rate decay scheme
        if config.use_lr_decay:
            if '_' not in config.lr_rate_decay:
                raise ValueError("Malformed learning_rate_decay")
            lrd_scheme, lrd_range = config.lr_rate_decay.split('_')

            if lrd_scheme not in lr_decay_map:
                raise ValueError("Unknown lr decay scheme {}".format(lrd_scheme))
            
            lrd_func = lr_decay_map[lrd_scheme]            
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                                            self.opt, 
                                            lrd_func(config),
                                            last_epoch=-1
                                        )
            # For each scheme, decay can happen every step or every epoch
            if lrd_range == 'epoch':
                self.lr_scheduler_epoch = lr_scheduler
            elif lrd_range == 'step':
                self.lr_scheduler_step = lr_scheduler
            else:
                raise ValueError("Unknown lr decay range {}".format(lrd_range))

        self.k = k
        self.model_name=config.model_name + self.k
        self.file_name = self.model_name + '.pth'
        self.model_file = os.path.join(config.output_dir, self.file_name)
        
        self.total_train_loss = []
        self.total_train_acc = []
        self.total_val_loss = []
        self.total_val_acc = []
        
        self.early_max_patience = config.early_max_patience
        
    # Load saved model
    def load_checkpoint(self):
        """
        Loads the trained model
        :param preds: None
        :return: None
        """           
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.opt = checkpoint['opt']
        self.opt.load_state_dict(checkpoint['opt_state'])
        self.total_train_loss = checkpoint['train_loss']
        self.total_train_acc = checkpoint['train_acc']
        self.total_val_loss = checkpoint['val_loss']
        self.total_val_acc = checkpoint['val_acc']
        self.epochs = checkpoint['epochs']
        
        
    # Save model
    def save_checkpoint(self):
        """
        Saves the trained model
        :param preds: None
        :return: None
        """      
        save_parameters = {'state_dict': self.model.state_dict(),
                           'opt': self.opt,
                           'opt_state': self.opt.state_dict(),
                           'train_loss' : self.total_train_loss,
                           'train_acc' : self.total_train_acc,
                           'val_loss' : self.total_val_loss,
                           'val_acc' : self.total_val_acc,
                           'epochs' : self.epochs}
        torch.save(save_parameters, self.model_file)        

    # Get the accuracy per batch
    def categorical_accuracy(self, preds, y):
        """
        Calculates the accuracy
        :param preds: predicted labels
        :param y: gold labels
        :return: batchwise accuracy
        """
        max_preds = preds.argmax(dim = 1, keepdim = True).squeeze(1) # get the index of the max probability
        correct = max_preds.eq(y)
        return correct.sum().item() / torch.FloatTensor([y.shape[0]])

    
    def train(self, model, iterator, optimizer, criterion):
        """
        Trains the given model
        :param model: lstm or cnn
        :param iterator: dataset iterator
        :param criterion: loss function
        :return: loss, accuracy
        """
        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for ((y, ac, at, X), v) in iterator:

            optimizer.zero_grad()
                        
            if self.config.train_type == 3:
                predictions = model(X, at, ac)
                gold = y
            else:
                predictions = model(X, at, None)
                gold = ac                

            gold = gold.squeeze(1)

            loss = criterion(predictions, gold)

            acc = self.categorical_accuracy(predictions, gold)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()


        return epoch_loss / self.train_dlen, epoch_acc / self.train_dlen    

    def evaluate(self, model, iterator, criterion):
        """
        Evaluates the given model
        :param model: lstm or cnn
        :param iterator: dataset iterator
        :param criterion: loss function
        :return: loss, accuracy, gold labels, predicted labels
        """    
        epoch_loss = 0
        epoch_acc = 0

        model.eval()
        
        gold_label = []
        pred_label = []
        
        with torch.no_grad():

            for ((y, ac, at, X), v) in iterator:

                if self.config.train_type == 3:
                    predictions = model(X, at, ac)
                    gold = y
                else:
                    predictions = model(X, at, None)
                    gold = ac                      

                gold = gold.squeeze(1)           
                
                gold_label.append(gold.data.cpu().numpy().tolist())
                pred_label.append(predictions.argmax(dim = 1, keepdim = True).squeeze(1).data.cpu().numpy().tolist())

                loss = criterion(predictions, gold)

                acc = self.categorical_accuracy(predictions, gold)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
        gold_list = [y for x in gold_label for y in x]
        pred_list = [y for x in pred_label for y in x]
        
        return epoch_loss / self.test_dlen, epoch_acc / self.test_dlen, gold_list, pred_list

    
    
    def fit(self):
        """
        Trains and evaluates the given model
        :param: 
        :return: 
        """            
        best_valid_loss = float('inf')
        counter = 0

        for epoch in tnrange(0, self.epochs):

            tqdm_t = tqdm(iter(self.train_dl), leave=False, total=self.train_dlen)
            tqdm_v = tqdm(iter(self.val_dl), leave=False, total=self.val_dlen)
            
            train_loss, train_acc = self.train(self.model, tqdm_t, self.opt, self.loss_fn)
            valid_loss, valid_acc, _,_ = self.evaluate(self.model, tqdm_v, self.loss_fn)
                
            if valid_loss < best_valid_loss:
                self.save_checkpoint()
                best_valid_loss = valid_loss
                counter=0
                self.logger.info("Best model saved!!!")
            else:
                counter += 1

            self.logger.info(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.4f}% | Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.4f}%')
        
            if counter >= self.early_max_patience: 
                self.logger.info("Training stopped because maximum tolerance reached!!!")
                break
    
    
    # Predict
    def predict(self):
        """
        Evaluates the given test data
        Writes the predicted results into a file
        :param: 
        :return: 
        """        
        evaluate = Evaluator(self.config, self.logger, self.model, self.dataloader, self.model_name)
        
        self.model.eval()
        tqdm_tst = tqdm(iter(self.test_dl), leave=False, total=self.test_dlen)      
        test_loss, test_acc, gold_list, pred_list = self.evaluate(self.model, tqdm_tst, self.loss_fn)
        self.logger.info(f'Test. Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.4f}%')
      
        # Get precision, recall and F1 score
        prec, rec, f1 = evaluate.prec_rec_f1(gold_list, pred_list)
        
        self.logger.info("Writing results")
        evaluate.write_results()
        
        return (test_acc, prec, rec, f1)
    
