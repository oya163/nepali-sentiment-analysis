'''
	Converts the .ann file into json format
	Author: Sandesh Timilsina
	Date: 02/06/2020

	How to run:
	- change the input_dir in the main function to give your directory 
	  which contains a list of channels
	python parser_new.py 
'''

import os
import argparse
import pandas as pd
import numpy as np
import json

count = 0
NER_CATEGORIES = ['PER','ORG','LOC','EVENT','DATE','NUM','MISC']
ASPECT_CATEGORIES = ['GENERAL', 'PROFANITY', 'VIOLENCE','SARCASM','FEEDBACK','OUTOFSCOPE']

def file_to_df(file):
    df = pd.read_csv(file, sep = "\t", header = None)
    df[['Aspect','Start','End']] = df[1].str.split(' ', expand=True)
    df['Keyword'] = df[2]
    df = df.rename(columns={0:'Term'})
    df = df.drop([1,2],axis=1)
    return df

def read_textfile(file):
    text_file = file.split('.')[0]+'.txt'
    f = open(text_file, 'r')
    text = f.readlines()
    f.close()
    return text

def read_textfile_asstring(file):
    text_file = file.split('.')[0]+'.txt'
    
    content = ""
    with open(text_file) as f:
        content = f.read()
    return content

def parser(df):
    output= []
    df['visited']= False
    for idx,row in df.iterrows():
        if row['Aspect']=='towards':
            df.loc[idx,'visited']=True
            
            res = {} # [NER, NER_CAT, ASPECT, ASP_CAT, Strength]

            aspect_term = row['Start'][-2:]
            ner_term = row['End'][-2:]
            entity_row,entity_idx = df[df['Term']==ner_term], df.index[df['Term']==ner_term].tolist()
            
            if not entity_row.empty and not df.loc[entity_idx[0],'visited']:
                df.loc[entity_idx[0],'visited']=True

            if len(entity_row)>=1:
                res['entity'] = entity_row['Keyword'].values.item()
                res['entity_from'] = int(entity_row['Start'].values.item()) 
                res['entity_to'] = int(entity_row['End'].values.item())
                res['entity_cat'] = entity_row['Aspect'].values.item()

            aspect_row, aspect_idx = df[df['Term']==aspect_term], df.index[df['Term']==aspect_term].tolist()
            strength_row, strength_idx = df[df['Start']==aspect_term], df.index[df['Start']==aspect_term].tolist()
            
            if not aspect_row.empty:
                df.loc[aspect_idx[0],'visited']=True
                
            if not strength_row.empty:
                df.loc[strength_idx[0],'visited']=True

            if len(aspect_row)>=1:
                res['aspect'] = aspect_row['Keyword'].values.item()
                res['aspect_from'] = int(aspect_row['Start'].values.item())
                res['aspect_to'] = int(aspect_row['End'].values.item())
                res['aspect_cat'] = aspect_row['Aspect'].values.item()
                if not strength_row.empty:
                    strength_row = [item for item in strength_row['End'] if item is not 'YES']
                    res['strength'] = strength_row[0]
            output.append(res)

    unvisted_df = df[df['visited']==False]
    for idx,row in unvisted_df.iterrows():
        if (df.loc[idx,'visited']==False):
            df.loc[idx,'visited']= True
            res = {}
            asp = row['Aspect']
            if asp in NER_CATEGORIES:
                res['entity'] = row['Keyword']
                res['entity_from'] = int(row['Start']) 
                res['entity_to'] = int(row['End'])
                res['entity_cat'] = row['Aspect']


            elif asp in ASPECT_CATEGORIES:
                aspect_term = row['Term'][-2:]
                strength_row  = unvisted_df[unvisted_df['Start']==aspect_term]
                if len(strength_row)>0:
                    strength_idx = df.index[df['Start']==aspect_term].tolist()
                    df.loc[strength_idx[0],'visited']= True
                    strength_row = [item for item in strength_row['End'] if item is not 'YES']
                    res['aspect'] = row['Keyword']
                    res['aspect_from'] = int(row['Start'])
                    res['aspect_to'] = int(row['End'])
                    res['aspect_cat'] = row['Aspect']
                    res['strength'] = strength_row[0]

                else:
                    res['aspect'] = row['Keyword']
                    res['aspect_from'] = int(row['Start'])
                    res['aspect_to'] = int(row['End'])
                    res['aspect_cat'] = row['Aspect']

            output.append(res)
    return output

def printdict(dict_data):
    print(json.dumps(dict_data,indent=4, ensure_ascii=False))
    
def get_splitpoint(text_list, all_text):
    out = [-1]*len(text_list)
    for i in range(len(text_list)):
        out[i]= all_text.rfind(text_list[i])
    return out

def get_filename(file):
    return file.split('.')[0]

def split_multicomments(input_dir, file, targeted_list, text, content, data):
    new_lines = get_splitpoint(text,content)
    new_lines.append(len(content)*2)
    status = [False]* len(targeted_list)
    for i in range(len(text)):
        channel, video_id = get_video_detail(input_dir)
        result_entry = {'channel' : channel,
                        'video_id' : video_id,
                        'comment_id': get_filename(file)+'__'+str(i+1),
                        'comment':text[i].strip(),
                        'tags':[]}

#         result_entry['comment'] = text[i].strip()
        for j in range(0,len(targeted_list)):
            if targeted_list[j].get('asp_from',0)!=0 and targeted_list[j]['asp_from'] <= new_lines[i+1] and not status[j]:
                status[j] = True
                if i>0:
                    targeted_list[j]['aspect_from'] -= new_lines[i]+i 
                    targeted_list[j]['aspect_to'] -= new_lines[i]+i
                    
                if 'entity_from' in targeted_list[j]:
                    targeted_list[j]['entity_from'] -= new_lines[i]+i
                    targeted_list[j]['entity_to'] -= new_lines[i]+i
                
                print()    
                result_entry.get('tags').append(targeted_list[j])
                
            elif targeted_list[j].get('entity_from',0)!=0 and targeted_list[j]['entity_from'] <= new_lines[i+1] and not status[j]:
                status[j] = True
                if i>0:
                    targeted_list[j]['entity_from'] -= new_lines[i]+i
                    targeted_list[j]['entity_to'] -= new_lines[i]+i
                
                if 'aspect_from' in targeted_list[j]:
                    targeted_list[j]['aspect_from'] -= new_lines[i]+i 
                    targeted_list[j]['aspect_to'] -= new_lines[i]+i
                    
                result_entry.get('tags').append(targeted_list[j])
        data.append(result_entry)
    return data

def get_video_detail(inp_dir):
    split_data = inp_dir.rsplit('/', 3)
    return split_data[2],split_data[3]
    
def process_single_video(input_dir):
    data = []
    for file in os.listdir(input_dir):
        f = os.path.join(input_dir,file)
        if f.endswith('ann') and os.stat(f).st_size != 0:
            df = file_to_df(f)
            text = read_textfile(f)
            targated_list = parser(df)
            result_entry = {}
            if (len(text)==1):
                channel, video_id = get_video_detail(input_dir)
                result_entry = { 'channel' : channel,
                                 'video_id' : video_id,
                                 'comment_id' : get_filename(file),
                                 'comment':text[0].strip(),
                                 'tags':targated_list
                               }
                data.append(result_entry)
            else:
                content = read_textfile_asstring(f)
                data = split_multicomments(input_dir, file, targated_list, text, content, data)
    return data

def json_dump(input_dir, channel, data):
    path = os.path.join(input_dir, channel+'.json')
    with open(path,'w') as json_file:
        json.dump(data, json_file)
    
def main():
    input_dir = '/home/sandesh/Desktop/brat/data/nepsa'
    final_json_data = []
    for channel in os.listdir(input_dir):
    	if channel.endswith('.json'):
    		continue

    	json_data = []
    	input_path = os.path.join(input_dir, channel)
    	for file in os.listdir(input_path):
    		vid_path = os.path.join(input_dir,channel,file)
    		data = process_single_video(vid_path)
    		json_data.extend(data)
    		final_json_data.extend(data)
    	json_dump(input_dir,channel, json_data)
    	print(channel, len(json_data))
    json_dump(input_dir,'all_channels', final_json_data)
    print("Total", len(final_json_data))
                
main()
