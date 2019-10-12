#!/usr/bin/env python3
    
"""
    Get the description of a Youtube video
    Author - Oyesh Mann Singh
    Date - 10/05/2019

    How to run:
        ./getinfo.py <YOUR API KEY> <YOUTUBE_VIDEO_ID>

    Description:
        Given Youtube Video ID, it extracts necessary
        fields like Video Title, number of comments,
        number of like/dislikes and much more
        and saves into a csv format file
"""

import sys
import pprint
import dateutil
import datetime as dt
import time
import csv

from apiclient.discovery import build

# Create and get Developer API Key from
# https://developers.google.com/
DEVELOPER_KEY = sys.argv[1]
youtube = build('youtube', 'v3', developerKey=DEVELOPER_KEY)


def main():
    ids = sys.argv[2]
    
    # Creat dictionary of Youtube Video Category ID
    id_map = {}
    with open("./category_map.txt") as f:
        for line in f:
            k, v = line.rstrip().split('\t')
            id_map[k] = v

    # Execute Youtube Video Data API
    snippet = youtube.videos().list(id=ids, part='snippet').execute()
    contentDetails = youtube.videos().list(id=ids, part='contentDetails').execute()
    statistics = youtube.videos().list(id=ids, part='statistics').execute()

    # Extract Snippet
    for result in snippet.get('items', []):
        channelTitle = result['snippet']['channelTitle']
        videoId = result['id']
        videoTitle = result['snippet']['title']
        uploadedDate = result['snippet']['publishedAt']
        categoryId = result['snippet']['categoryId']
        category = id_map[categoryId]
    
    # Extract Content Details
    for result in contentDetails.get('items', []):
        videoLength = result['contentDetails']['duration']
        # Convert ISO 8601 to human readable
        try:
            # For video less than an hour
            parsedTime = dt.datetime.strptime(videoLength, "PT%MM%SS")
            myFormat = "%M:%S"
        except:
            # For video more than an hour
            parsedTime = dt.datetime.strptime(videoLength, "PT%HH%MM%SS")
            myFormat = "%H:%M:%S"
        
        videoLength = parsedTime.strftime(myFormat)

    # Extract Statistics
    for result in statistics.get('items', []):
        viewCount = result['statistics']['viewCount']
        likeCount = result['statistics']['likeCount']
        dislikeCount = result['statistics']['dislikeCount']
        commentCount = result['statistics']['commentCount']
        
    # Create list of fields
    row = [category, channelTitle, videoId, videoTitle, videoLength, 
            uploadedDate, viewCount, commentCount, likeCount, dislikeCount]

    # Write into csv
    with open('info.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    f.close()


if __name__=="__main__":
    main()
