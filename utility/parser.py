#!/usr/bin/env python

'''
    Parse the annotation file
    Author : Sandesh Timilsina
    Date: 10/22/2019
'''

import pandas as pd

class Parser():
    entity = []
    entity_class = []
    attribute = []
    attribute_class = []
    label = []
    
    def file_to_df(self, file):
        df = pd.read_csv(file, sep = "\t", header = None)
        df[['Aspect','Start','End']] = df[1].str.split(' ', expand=True)
        df['Keyword'] = df[2]
        df = df.rename(columns={0:'Term'})
        df = df.drop([1,2],axis=1)
        return df

    def parse(self, df):
        for idx,row in df.iterrows():
            if row['Aspect'] == 'AspectTerm':
                strength,category,keyword = df['End'][(df['Start']==row['Term']) & (df['Aspect']=='Strength')].item(), \
                            df['End'][(df['Start']==row['Term']) & (df['Aspect']=='AspectCategory')].item(),row['Keyword']
#                 print(strength,category,keyword)
                self.label.append(strength)
                self.attribute_class.append(category)
                self.attribute.append(keyword)
            elif row['Aspect'] == "Target": 
                category, ner = df['End'][df['Start']==row['Term']].item(), row['Keyword']
                self.entity.append(ner)
                self.entity_class.append(category)
#                 print(category,ner)
            else:
                continue

    def get_parser_file_1(self):
        max_len = max(len(entity), len(entity_class), len(attribute), len(attribute_class), len(label))
        res = []
        for i in range(max_len):
            res.append([entity[i] if entity[i] else None,
                        entity_class[i] if entity_class[i] else None,
                        attribute[i] if attribute[i] else None,
                        attribute_class[i] if attribute_class[i] else None,
                        label[i] if label[i] else None])
        return res;
