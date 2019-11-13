#!/bin/bash

# How to run
# bash anntoconll.sh ../data/nepsa/Avenues_Khabar/0S8tX4eRa6M/

# Input data dir
DATA_DIR=$1

# For all *.txt files
# run the script
for filename in $DATA_DIR/*.txt; do
	echo "Processing: $filename"
	python anntoconll.py $filename
done
