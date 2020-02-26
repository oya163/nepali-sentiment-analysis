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
    def __init__(self, config, k):
        self.root_path = os.path.join(config.root_path, k)
        self.batch_size = config.batch_size
        self.device = config.device
        
        self.txt_field = data.Field(tokenize=list, use_vocab=True, unk_token='<unk>', batch_first=True)
        self.at_field = data.Field(tokenize=list, use_vocab=True, unk_token='<unk>', batch_first=True)
        self.ac_field = data.Field(batch_first=True)
        self.label_field = data.Field(batch_first=True)
        
        self.fields = (('LABEL', self.label_field), ('TERM', self.at_field), ('TEXT', self.txt_field))
        
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
        self.label_field.build_vocab(self.train_ds.LABEL)
        
        self.vocab_size = len(self.txt_field.vocab)
        self.tagset_size = len(self.label_field.vocab)
        self.at_size = len(self.at_field.vocab)
        
        self.weights = self.txt_field.vocab.vectors
        
        if config.verbose:
            self.print_stat()

        
    def train_ds(self):
        return self.train_ds
    
    def val_ds(self):
        return self.val_ds    
    
    def test_ds(self):
        return self.test_ds
    
    def txt_field(self):
        return self.txt_field
    
    def label_field(self):
        return self.label_field    
    
    def vocab_size(self):
        return self.vocab_size

    def tagset_size(self):
        return self.tagset_size

    def weights(self):
        return self.weights
    
    def print_stat(self):
        """
        Prints the data statistics
        """           
        print('Length of training dataset = ', self.train_ds)
        print('Length of testing dataset = ', self.test_ds)
        print('Length of validation dataset = ', self.val_ds)
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
                                            shuffle=True)

        return train_iter, val_iter, test_iter
