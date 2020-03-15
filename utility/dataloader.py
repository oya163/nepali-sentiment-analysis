#!/usr/bin/env python3

'''
    CSV Dataloader
    Author: Oyesh Mann Singh
    Data format:
        "LABEL", "ASPECT_TERM", "TEXT"

'''

import io
import os
import logging
import numpy as np

import torch
import torchtext
from torchtext import data
from torchtext import vocab
from torchtext import datasets
from torchtext.datasets import SequenceTaggingDataset

from uniseg.graphemecluster import grapheme_clusters

class Dataloader():
    def __init__(self, config, k=1):
        
        self.root_path = os.path.join(config.root_path, k)
        self.batch_size = config.batch_size
        self.device = config.device
        
        self.txt_field = data.Field(tokenize=self.tokenizer, use_vocab=True, unk_token='<unk>', batch_first=True)
        self.at_field = data.Field(tokenize=self.tokenizer, use_vocab=True, unk_token='<unk>', batch_first=True)
        self.ac_field = data.Field(batch_first=True, unk_token=None, pad_token=None)
        self.ss_field = data.Field(batch_first=True, unk_token=None, pad_token=None)
              
        self.fields = (('SS', self.ss_field), ('ASPECT', self.ac_field), 
                       ('TERM', self.at_field), ('TEXT', self.txt_field))            

        self.train_ds, self.val_ds, self.test_ds = data.TabularDataset.splits(path=self.root_path, 
                                                    format='csv', 
                                                    train='train.txt', 
                                                    validation='val.txt', 
                                                    test='test.txt',                          
                                                    fields=self.fields)

        self.embedding_dir = config.emb_dir
        self.vec = vocab.Vectors(name=config.emb_file, cache=self.embedding_dir)

        self.txt_field.build_vocab(self.train_ds.TEXT, self.test_ds.TEXT, self.val_ds.TEXT, max_size=None, vectors=self.vec)
        self.at_field.build_vocab(self.train_ds.TERM, self.test_ds.TERM, self.val_ds.TERM, max_size=None, vectors=self.vec)
        self.ac_field.build_vocab(self.train_ds.ASPECT)
        self.ss_field.build_vocab(self.train_ds.SS)
                    
        self.vocab_size = len(self.txt_field.vocab)
        self.at_size = len(self.at_field.vocab)
        self.ac_size = len(self.ac_field.vocab)
        self.ss_size = len(self.ss_field.vocab)
        self.tagset_size = self.ac_size
        
        # One hot encoding for aspect category
        if config.train_type == 3:
            self.one_hot_aspect = torch.eye(self.ac_size, dtype=torch.float, requires_grad=True)
            self.tagset_size = self.ss_size    
        
        self.weights = self.txt_field.vocab.vectors
        
        if config.verbose:
            self.print_stat()

    def tokenizer(self, x):
        return x.split()        
    
    def train_ds(self):
        return self.train_ds
    
    def val_ds(self):
        return self.val_ds    
    
    def test_ds(self):
        return self.test_ds

    def train_ds_size(self):
        return len(self.train_ds)
    
    def val_ds_size(self):
        return len(self.val_ds)
    
    def test_ds_size(self):
        return len(self.test_ds)    
    
    def txt_field(self):
        return self.txt_field
    
    def ss_field(self):
        return self.ss_field    
    
    def vocab_size(self):
        return self.vocab_size

    def tagset_size(self):
        return self.tagset_size

    def at_size(self):
        return self.at_size
    
    def ac_size(self):
        return self.ac_size
    
    def weights(self):
        return self.weights
    
    def print_stat(self):
        """
        Prints the data statistics
        """           
        print('Length of training dataset = ', len(self.train_ds))
        print('Length of testing dataset = ', len(self.test_ds))
        print('Length of validation dataset = ', len(self.val_ds))
        print('Length of text vocab (unique words in dataset) = ', self.vocab_size)
        print('Length of label vocab (unique tags in labels) = ', self.tagset_size)
    
    def load_data(self, batch_size, shuffle=True):
        """
        Loads the data
        :param batch_size: batch_size
        :param shuffle: shuffle
        :return: train, val, test iterator
        """        
        train_iter, val_iter, test_iter = data.BucketIterator.splits(datasets=(self.train_ds, self.val_ds, self.test_ds), 
                                            batch_sizes=(batch_size, batch_size, batch_size), 
                                            sort_key=lambda x: len(x.TEXT), 
                                            device=self.device, 
                                            sort_within_batch=True, 
                                            repeat=False,
                                            shuffle=False)

        return train_iter, val_iter, test_iter
