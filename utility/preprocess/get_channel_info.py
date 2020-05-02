# -*- coding: utf-8 -*-

# Sample Python code for youtube.search.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python

# Get most subscribed youtube video channel in Nepal
# https://vidooly.com/top-youtube-channels/NP/mostsubscribed

# How to run:
# python3 get_channel_info.py <CHANNEL_ID>
# python3 get_channel_info.py UCNR1KcWXj7zpWQFJtU3ddYg

# How to get channelid
# Play any video of desired channel
# Right click and View Page Source
# Search for channelid


import os
import sys
import json


import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]

CHANNEL_ID = sys.argv[1]

def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    # os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "MY_YOUTUBE_CLIENT.json"

    # Get credentials and create an API client
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_console()
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)

    request = youtube.search().list(
        part="snippet",
        channelId=CHANNEL_ID,
        maxResults=50,
        order="viewCount",
        prettyPrint=True
    )
    response = request.execute()

    channelTitle = response["items"][0]["snippet"]["channelTitle"]
    
    filename = channelTitle+'_'+CHANNEL_ID+'.json'
    filepath = '../data/channels/'+filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=4)
        
    print("Written in :", filepath)

if __name__ == "__main__":
    main()