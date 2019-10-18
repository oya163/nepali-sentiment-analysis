import os

'''
    Author: Sandesh Timilsina
    Date: 10/15/2019

    How to run:
        python <filename.py>
    
    Description:
        Given a list of video ids, this script downloads
        all the comments from the youtube.

    Requirement:
        Install youtube-comment-scrapper-cli.
'''


urls= ['RxfWgvB-Zcc',
'x5zHAu_UULI',
'7aSSjVrkYmI',
'QeZ2WYLJMmg',
'l6SAoNM_b6U',
'7I4HfcAnYzA',
'J0CtOeLbIYk',
'SjlZw4iVq2Y',
'qKFQ57sHdD8',
'zZ8Y5ncN3Po',
'AQBk02D5PNE']

for url in urls:
    print(url,end=" ")
    os.system("youtube-comment-scraper -o "+url+".json -f json "+url)
    print("Done")
