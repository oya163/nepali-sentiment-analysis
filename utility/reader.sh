#/bin/bash

# Scraps all the comments of given video

# How to run
# bash reader.sh video_list.txt

# Reference - https://github.com/philbot9/youtube-comment-scraper-cli
# youtube-comment-scraper -c -o ../data/corpus/comments/$filename $videoid
# -c = collapse replies and treat them the same as regular comments
# -o = output file

# Input video_list file
FILE=$1

# For all *.txt files
# run the script
while IFS= read -r videoid; do
    filename="../data/corpus/all_comments/"${videoid}".json"
    if [ ! -f "$filename" ]; then
        echo "Writing in "$filename
        youtube-comment-scraper -c -o $filename $videoid
    else
        echo "${filename} ..... Already exists!!!"
    fi
done < "$FILE"
