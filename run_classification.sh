#!/bin/bash

# Run LSTM classifier
python main.py -t 3 -k 5 -m lstm -d cuda:1 &
python main.py -t 4 -k 5 -m lstm -d cuda:1 &

# Run CNN classifier
python main.py -t 3 -k 5 -m cnn -d cuda:1 &
python main.py -t 4 -k 5 -m cnn -d cuda:1 &