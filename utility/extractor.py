#!/usr/bin/env python3

'''
    Extractor of commentText
'''

import os
import json
import sys
import emoji

def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        comments = json.load(f)
        
    for each in enumerate(comments):
        msg = emoji.demojize(each[1]['commentText'])+'\n'
        filename = str(each[0])+'_ravi'
        txtfile = filename + '.txt'
        annfile = filename + '.ann'
        with open(txtfile, 'w', encoding='utf-8') as f, open(annfile, 'a', encoding='utf-8') as af:
            f.write(msg)


if __name__ == "__main__":
    main()
