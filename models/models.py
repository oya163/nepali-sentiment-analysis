'''
    Models
    Author: Oyesh Mann Singh
    Model available:
        LSTM
        CNN
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, config, dataloader):
        super().__init__()
        self.config = config
        self.bidirectional = config.bidirection
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.vocab_size = dataloader.vocab_size
        self.tagset_size = dataloader.tagset_size
        self.embedding_dim = config.embedding_dim
        self.train_type = config.train_type
        
        if config.pretrained:
            self.embedding = nn.Embedding.from_pretrained(dataloader.weights)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)   
            
            
        if self.config.train_type == 3:
            self.ac_size = dataloader.ac_size
#             print("AC size", self.ac_size)
            self.ac_embeddings = nn.Embedding(self.ac_size, config.embedding_dim)
#             self.embedding_dim = config.embedding_dim + self.ac_size
#             self.one_hot_aspect = np.eye(self.ac_size)
#             self.one_hot_aspect = torch.from_numpy(self.one_hot_aspect).float()            
#             self.one_hot_aspect = torch.eye(self.ac_size, dtype=torch.float, requires_grad=True)
#             print("One hot aspect", self.one_hot_aspect)
#             print("One hot aspect shape", self.one_hot_aspect.shape)
            
            
        
        self.lstm = nn.LSTM(self.embedding_dim, 
                           self.hidden_dim, 
                           num_layers=self.num_layers, 
                           bidirectional=self.bidirectional)
#                           dropout=0.5)
        
        self.fc = nn.Linear(self.hidden_dim * 2, self.tagset_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, text, at, ac):
        #text = [batch size, sent len]
#         print("Text shape", text.shape)
#         print("Aspect term shape", at.shape)
        
        text = text.permute(1, 0)
        at = at.permute(1, 0)
        
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
        at_emb = self.embedding(at)
        
#         print("Embedded shape ", embedded.shape)        
#         print("Aspect term shape ", at_emb.shape)        
        
        # Only concatenate text and aspect term
        if self.train_type == 2:
            embedded = torch.cat((embedded, at_emb), dim=0)

            
        # Only concatenate text, aspect term and category
        if self.train_type == 3:
            embedded = torch.cat((embedded, at_emb), dim=0)
#             print("Aspect before permute", ac.shape)
            ac = ac.permute(1, 0)
#             print("Aspect after permute", ac.shape)
#             print("Aspect shape", ac.shape)
            ac_emb = self.ac_embeddings(ac)
            
#             print("Embedded shape after first concatenation", embedded.shape)
#             print("Aspect embd shape", ac_emb.shape)
            seq_len = embedded.shape[1]
#             ac_emb = ac_emb.view(self.ac_size, seq_len, self.ac_size)
#             print("Aspect embd shape", ac_emb.shape)
            
            embedded = torch.cat([embedded, ac_emb], dim=0)
#             print("Embeddings after all concatenation", embedded)
#             print("Embedded shape after all concatenation", embedded.shape)

        #embedded = [sent len (embedded+aspect_emb), batch size, emb dim]
        
        embedded, (hidden, cell) = self.lstm(self.dropout(embedded))
#        embedded, (hidden, cell) = self.lstm(embedded)
                
#         embedded = [batch size, sent_len, num_dim * hidden_dim]
#         hidden = [num_dim, sent_len, hidden_dim]

#        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
#         print("hidden shape ", hidden.shape)
                
#         hidden = [batch size, hid dim * num directions]
        final = F.softmax(self.fc(hidden), dim=-1)
        
        return final



class CNN(nn.Module):
    def __init__(self, config, dataloader):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.vocab_size = dataloader.vocab_size
        self.tagset_size = dataloader.tagset_size
        self.embedding_dim = config.embedding_dim
        self.train_type = config.train_type
        self.num_filters = config.num_filters
        
        # Because filter_sizes are passed as string
        self.filter_sizes = [int(x) for x in config.filter_sizes.split(',')]
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        if self.config.train_type == 3:
            self.ac_size = dataloader.ac_size
            self.ac_embeddings = nn.Embedding(self.ac_size, self.embedding_dim)
    
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = self.num_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in self.filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, self.tagset_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, text, at, ac):
        
        #text = [sent len, batch size]
#         print("Shape of text", text.shape)
#         print("Shape of aspect", aspect.shape)
        
        #text = [batch size, sent len]
        
#         text = text.permute(1, 0)
#         aspect = aspect.permute(1, 0)

        #text = [sent len, batch size]
        
#         print("Shape of text", text.shape)
#         if self.train_type > 1:
#             print("Shape of aspect", aspect.shape)        
        
        embedded = self.embedding(text)
        at_emb = self.embedding(at)
        
        if self.train_type == 2:
            embedded = torch.cat((embedded, at_emb), dim=1)

        # Only concatenate text, aspect term and category
        if self.train_type == 3:
            embedded = torch.cat((embedded, at_emb), dim=1)
#             print("Aspect before permute", ac.shape)
#             ac = ac.permute(1, 0)
#             print("Aspect after permute", ac.shape)

#             print("Embedding shape", embedded.shape)
#             print("Aspect shape", ac.shape)
            ac_emb = self.ac_embeddings(ac)
#             print("Aspect emb shape", ac_emb.shape)
            
#             print("Embedded shape after first concatenation", embedded.shape)
#             print("Aspect embd shape", ac_emb.shape)
            seq_len = embedded.shape[1]
#             ac_emb = ac_emb.view(self.ac_size, seq_len, self.ac_size)
#             print("Aspect embd shape", ac_emb.shape)
            
            embedded = torch.cat([embedded, ac_emb], dim=1)
        
        
#         print("Shape of embedded", embedded.shape)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
#         print("Shape of embedded after squeeze", embedded.shape)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        #conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        conved = [F.relu(conv(self.dropout(embedded))).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
#         print("Shape of pooled", pooled[0].shape)
        
        #pooled_n = [batch size, n_filters]
        
        #cat = self.dropout(torch.cat(pooled, dim = 1))
        cat = torch.cat(pooled, dim = 1)
#         print("Shape of cat", cat.shape)

        #cat = [batch size, n_filters * len(filter_sizes)]
#         print("Shape of final", final.shape)
        
        final = F.softmax(self.fc(cat), dim=-1)
        
        return final
