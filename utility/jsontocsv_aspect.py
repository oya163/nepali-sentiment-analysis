'''
    Create json to csv aspect format
    Author: Sandesh Timilsina
    Date: 02/17/2020
    
    Example:
    "sentiment strength", "aspect category", "aspect term", "complete sentence"
    
    How to run:
    >> python jsontocsv_aspect.py -file ../../brat/data/Raw/all_channels.json 
'''
import os
import sys
import argparse
import json
import csv
import pandas as pd

def read_json_file(file):
    with open(file) as f:
        data = json.load(f)
    return data

def create_df(columns):
    return pd.DataFrame(columns = columns)

def prepare_aspect_format(json_data):
    print("Preparing aspect format")
    df = create_df(['strength','asp_category','asp_team','entity_category','entity_term','comment'])
    for item in json_data:
        comment = item['comment'].strip('\n')
        tags = item['tags']
        for tag in tags:
            if tag and not tag.get('strength',None)==None:
                strength = tag.get('strength',None)
                asp_category = tag.get('aspect_cat',None)
                asp_term = tag.get('aspect', None)
                entity_category = tag.get('entity_cat',None)
                entity_term = tag.get('entity', None)
                df.loc[len(df)] = [strength, asp_category, asp_term, entity_category, entity_term, comment]
    return df

def write_df_to_file(df,output_file_path):
    print("Writing to file....")
    df.to_csv(output_file_path, header=False, index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    print("Writing to file completed")

def main(argv):
    parser =  argparse.ArgumentParser(add_help=True, description=('Filter json file'))
    parser.add_argument('-file', default="../../brat/data/nepsa/all_channels.json", help='json file path')
    args = parser.parse_args(argv)
    input_file = args.file

    data = read_json_file(input_file)
    df = prepare_aspect_format(data)
    output_file = input_file.rpartition('/')[0] + "/aspect_format.csv" 
    write_df_to_file(df, output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
