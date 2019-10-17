import os

<<<<<<< HEAD
urls= ['ATEiz1N58D0',
'sQ45_FoSV0k',
'1nXx-0EK-ns',
'rCPPVS_dwlw',
'OJWwMbLO0DA',
'4jOmSfBLRlk',
'xah2_pUBkIc',
'FcLe08Rtz_M']
=======
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
>>>>>>> 73ade3c4e6d43b9cb4c04627805e06fd928b8e43

        
for url in urls:
    print(url,end=" ")
    os.system("youtube-comment-scraper -o "+url+".json -f json "+url)
    print("Done")
