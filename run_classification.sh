#!/bin/bash

'''
    This script will run lstm and cnn
    trainer for train_type 3 and 4
    with 5-fold cross validation and
    produces results in corresponding
    folders with logs.
    
    For final results, please check
    ./logs directory and respective filename
    
    Log file format:
        complete_<MODEL>_<TRAIN_TYPE>.log
        
    How to run:
        bash run_classification.sh
'''

KFOLD=5
CUDA=cuda:1
DATA_DIR=./data/nepcls
INPUT_PATH=${DATA_DIR}/csv
ROOT_PATH=${DATA_DIR}/kfold

DATA_FILENAME=ss_ac_at_txt_unbal
DATA_FILE="${DATA_FILENAME}.csv"
DATA_PATH=${INPUT_PATH}/${DATA_FILE}

python utility/splitter.py -c -i ${DATA_PATH} -o ${ROOT_PATH} -k ${KFOLD} -v

for MODEL in lstm cnn
do
    for TRAIN_TYPE in 3 4
        do 
            MODEL_NAME="${DATA_FILENAME}_${MODEL}_${TRAIN_TYPE}"
            
            python main.py -r ${ROOT_PATH} -t ${TRAIN_TYPE} \
                           -k ${KFOLD} -n ${MODEL_NAME} \
                           -m ${MODEL} -d ${CUDA}
            
            exit 1
        done        
done
