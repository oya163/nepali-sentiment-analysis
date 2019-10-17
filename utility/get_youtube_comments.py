import os

urls= ['ATEiz1N58D0',
'sQ45_FoSV0k',
'1nXx-0EK-ns',
'rCPPVS_dwlw',
'OJWwMbLO0DA',
'4jOmSfBLRlk',
'xah2_pUBkIc',
'FcLe08Rtz_M']

        
for url in urls:
    print(url,end=" ")
    os.system("youtube-comment-scraper -o "+url+".json -f json "+url)
    print("Done")
