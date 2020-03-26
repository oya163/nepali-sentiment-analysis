#!/bin/bash

# Run LSTM classifier
python main.py -t 3 -k 5 -m lstm
python main.py -t 4 -k 5 -m lstm

# Run CNN classifier
python main.py -t 3 -k 5 -m cnn
python main.py -t 4 -k 5 -m cnn