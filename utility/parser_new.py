import os
import argparse
import pandas as pd
import numpy as np

NER_CATEGORIES = ['PER','ORG','LOC','EVENT','DATE','NUM','MISC']
ASPECT_CATEGORIES = ['GENERAL', 'PROFANITY', 'VIOLENCE','SARCASM','FEEDBACK','OUTOFSCOPE']

def file_to_df(file):
    df = pd.read_csv(file, sep = "\t", header = None)
    df[['Aspect','Start','End']] = df[1].str.split(' ', expand=True)
    df['Keyword'] = df[2]
    df = df.rename(columns={0:'Term'})
    df = df.drop([1,2],axis=1)
    return df

def parser(df):
    output= []
    df['visited']= False
    for idx,row in df.iterrows():
        if row['Aspect']=='towards':
            df.loc[idx,'visited']=True

            res = [None]*5 # [NER, NER_CAT, ASPECT, ASP_CAT, Strength]

            aspect_term = row['Start'][-2:]
            ner_term = row['End'][-2:]
            entity_row,entity_idx = df[df['Term']==ner_term], df.index[df['Term']==ner_term].tolist()
            df.loc[entity_idx[0],'visited']=True

            if len(entity_row)>=1:
                res[:2] = entity_row['Keyword'].item(),entity_row['Aspect'].item()
            aspect_row, aspect_idx = df[df['Term']==aspect_term], df.index[df['Term']==aspect_term].tolist()
            strength_row, strength_idx = df[df['Start']==aspect_term], df.index[df['Start']==aspect_term].tolist()

            df.loc[aspect_idx[0],'visited']=True
            df.loc[strength_idx[0],'visited']=True
            if len(aspect_row)>=1:
                res[2:] = aspect_row['Keyword'].item(),aspect_row['Aspect'].item(),strength_row['End'].item()
            output.append(res)
    print(res)
    unvisted_df = df[df['visited']==False]
    for idx,row in unvisted_df.iterrows():
        if (df.loc[idx,'visited']==False):
            df.loc[idx,'visited']= True
            res = [None]*5
            asp = row['Aspect']
            if asp in NER_CATEGORIES:
                res[:2] = row['Keyword'],row['Aspect']

            elif asp in ASPECT_CATEGORIES:
                aspect_term = row['Term'][-2:]
                strength_row  = unvisted_df[unvisted_df['Start']==aspect_term]
                strength_idx = df.index[df['Start']==aspect_term].tolist()
                df.loc[strength_idx[0],'visited']= True
                res[2:] = row['Keyword'],row['Aspect'],strength_row['End'].item()
            output.append(res)
    return output


def main():
    input_dir = '/home/sandesh/Desktop/brat/data/nepali_data/Avenues_Khabar/0S8tX4eRa6M/'
    for file in os.listdir(input_dir):
        f = os.path.join(input_dir,file)
        if f.endswith('ann') and os.stat(f).st_size != 0:
            df = file_to_df(f)
            targated_list = parser(df)
            print(targated_list)
main()
