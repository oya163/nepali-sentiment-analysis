'''
    Configuration Parser
    Author: Oyesh Mann Singh
    Date 10/15/2019

'''

import os
import logging
from configparser import ConfigParser

class Configuration(ConfigParser):
    def __init__(self, config_file, logger, args):
        super().__init__()
        config = ConfigParser(allow_no_value=True)
        config.read(config_file)
        self._config = config
        self.config_file = config_file
        self.logger = logger
        self.args = args

        self.logger.info("Configuration file loaded!!!")

        for section in config.sections():
            self.logger.info("*******************************"+section)
            for k, v in config.items(section):
                self.logger.info("{} : {}".format(k,v))

    # ------------------DATA
    @property
    def data_file(self):
        return self._config.get('DATA', 'data_file')    
    
    @property
    def root_path(self):
        return os.path.join(self._config.get('DATA', 'root_path'), self.args.model_name)
    
    @property
    def shuffle(self):
        return self._config.getboolean('DATA', 'shuffle')
    
    # ------------------EMBEDDINGS
    @property
    def pretrained(self):
        return self._config.getboolean('EMBEDDINGS', 'pretrained')
    
    @property
    def emb_dir(self):
        return self._config.get('EMBEDDINGS', 'emb_dir')
    
    @property
    def emb_file(self):
        return self._config.get('EMBEDDINGS', 'emb_file')
    
    @property
    def embedding_dim(self):
        return self._config.getint('EMBEDDINGS', 'embedding_dim')
    
    @property
    def embed_finetune(self):
        return self._config.getboolean('EMBEDDINGS', 'embed_finetune')  
    
    @property
    def char_pretrained(self):
        return self._config.getboolean('EMBEDDINGS', 'char_pretrained')  
    
    @property
    def char_emb_file(self):
        return self._config.get('EMBEDDINGS', 'char_emb_file')
    
    @property
    def graph_emb_file(self):
        return self._config.get('EMBEDDINGS', 'graph_emb_file')
    
    @property
    def char_dim(self):
        return self._config.getint('EMBEDDINGS', 'char_dim')    
    
    # ------------------OUTPUT_DIR
    @property
    def output_dir(self):
        return self._config.get('OUTPUT_DIR', 'output_dir')
    
    @property
    def results_dir(self):
        return self._config.get('OUTPUT_DIR', 'results_dir')    
    
    # ------------------TRAIN
    @property
    def batch_size(self):
        return self._config.getint('TRAIN', 'batch_size')
    
    @property
    def epochs(self):
        return self._config.getint('TRAIN', 'epochs')
    
    @property
    def early_max_patience(self):
        return self._config.getint('TRAIN', 'early_max_patience')
    
    @property
    def log_interval(self):
        return self._config.getint('TRAIN', 'log_interval')    
    
    # ------------------OPTIMIZER
    @property
    def adam(self):
        return self._config.getboolean("OPTIM", "adam")

    @property
    def learning_rate(self):
        return self._config.getfloat("OPTIM", "learning_rate")

    @property
    def weight_decay(self):
        return self._config.getfloat("OPTIM", "weight_decay")

    @property
    def momentum(self):
        return self._config.getfloat("OPTIM", "momentum")

    @property
    def clip_max_norm_use(self):
        return self._config.getboolean("OPTIM", "clip_max_norm_use")

    @property
    def clip_max_norm(self):
        return self._config.get("OPTIM", "clip_max_norm")

    @property
    def use_lr_decay(self):
        return self._config.getboolean("OPTIM", "use_lr_decay")

    @property
    def lr_rate_decay(self):
        return self._config.get("OPTIM", "lr_rate_decay")
    
    @property
    def learning_rate_warmup_steps(self):
        return self._config.getfloat("OPTIM", "learning_rate_warmup_steps")
    
    @property
    def min_lrate(self):
        return self._config.getfloat("OPTIM", "min_lrate")

    @property
    def max_patience(self):
        return self._config.getint("OPTIM", "max_patience")
    
    # ------------------MODEL
    @property
    def bidirection(self):
        return self._config.getboolean('MODEL', 'bidirection')
    
    @property
    def num_layers(self):
        return self._config.getint('MODEL', 'num_layers')
    
    @property
    def hidden_dim(self):
        return self._config.getint('MODEL', 'hidden_dim')
    
    @property
    def dropout_embed(self):
        return self._config.getfloat('MODEL', 'dropout_embed')
    
    @property
    def dropout(self):
        return self._config.getfloat('MODEL', 'dropout')

    @property
    def num_filters(self):
        return self._config.getint('MODEL', 'num_filters')    
    
    @property
    def filter_sizes(self):
        return self._config.get('MODEL', 'filter_sizes')        
    
    # ------------------EVALUATION
    @property
    def average(self):
        return self._config.get('EVALUATION', 'average')
       