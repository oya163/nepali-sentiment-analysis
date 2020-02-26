'''
    Models
    Author: Oyesh Mann Singh
    Model available:
        LSTM
        CNN
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, config, dataloader):
        super().__init__()
        
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
        
        self.lstm = nn.LSTM(self.embedding_dim, 
                           self.hidden_dim, 
                           num_layers=self.num_layers, 
                           bidirectional=self.bidirectional,
                           dropout=0.5)
        
        self.fc = nn.Linear(self.hidden_dim * 2, self.tagset_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, text, aspect):
        #text = [batch size, sent len]
        
        text = text.permute(1, 0)
        aspect = aspect.permute(1, 0)

        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
        aspect_emb = self.embedding(aspect)
        
        if self.train_type == 2:
            embedded = torch.cat((embedded, aspect_emb), dim=0)
    
        #embedded = [sent len (embedded+aspect_emb), batch size, emb dim]
        
#         embedded, (hidden, cell) = self.lstm(self.dropout(embedded))
        embedded, (hidden, cell) = self.lstm(embedded)
                
#         embedded = [batch size, sent_len, num_dim * hidden_dim]
#         hidden = [num_dim, sent_len, hidden_dim]

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
#         print("hidden shape ", hidden.shape)
                
#         hidden = [batch size, hid dim * num directions]
        final = self.fc(hidden)

        return final



class CNN(nn.Module):
    def __init__(self, config, dataloader):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.vocab_size = dataloader.vocab_size
        self.tagset_size = dataloader.tagset_size
        self.embedding_dim = config.embedding_dim
        self.train_type = config.train_type
        self.num_filters = config.num_filters
        
        # Because filter_sizes are passed as string
        self.filter_sizes = [int(x) for x in config.filter_sizes.split(',')]    
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = self.num_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in self.filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, self.tagset_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, text, aspect):
        
        #text = [sent len, batch size]
#         print("Shape of text", text.shape)
#         print("Shape of aspect", aspect.shape)
        
        #text = [batch size, sent len]
        
#         text = text.permute(1, 0)
#         aspect = aspect.permute(1, 0)

        #text = [sent len, batch size]
        
#         print("Shape of text", text.shape)
#         print("Shape of aspect", aspect.shape)        
        
        embedded = self.embedding(text)
        aspect_emb = self.embedding(aspect)
        
        if self.train_type == 2:
            embedded = torch.cat((embedded, aspect_emb), dim=1)
        
#         print("Shape of embedded", embedded.shape)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
#         print("Shape of embedded after squeeze", embedded.shape)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
#         print("Shape of conved", conved[0].shape)
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
#         print("Shape of pooled", pooled[0].shape)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
#         print("Shape of cat", cat.shape)

        #cat = [batch size, n_filters * len(filter_sizes)]
        final = self.fc(cat)
#         print("Shape of final", final.shape)
        
        return final    