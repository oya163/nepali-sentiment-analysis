# Nepali Sentiment Analysis

With an increase in internet access in Nepal and ease of typing in Nepali, this language has become prevalent on social media platforms. However, there is a lack of fine-grained sentiment analysis for comments written in Nepali. In this project, we aim to analyze the YouTube comments written in Nepali, both code-mixed and code-switched. As a first step, we are creating a Named Targeted Aspect Based Sentiment Analysis Dataset from comments extracted from popular Nepali YouTube videos under the News & Politics category. This dataset can be used for empirical studies on multilingual training, domain adaptation and vector-space word mapping techniques. Hence, this dataset can play a major role in identifying abusive comments in Nepali texts.


## Authors

- Oyesh Singh
- Sandesh Timilsina


### Research roadmap

- [X] Dataset collection
- [X] Dataset annotation
- [X] Model comparison


Folder Structure for Output:
    
    .
    ├── ...
    ├── utility					# Folder that contains all the utility files
    │   ├── get_info.py                     		# Gets all the details from a given video id
    │   ├── get_youtube_comments.py         		# Gets all the youtube comments from a video id 
    │   ├── random_sample.py                		# Performs different pre-processing (random sampling, demojify) 
    │   └── unitag_to_conll.py                  	# Preprocessing after running unitag (separate comment text from POS tags etc) 
    │
    ├── youtube                    	    	# Folder that contains input-output files 
    │	├── raw                				# Input Folder (contains comments scrapped from different channels)
    │	│   ├── channel 1					# Name of the channel 
    │	│   │   ├── json file 1                				# Json file that contains comments from specific video 
    │	│   │	├── json file 2						# Json file that contains comments from specific video
    │	│   │	└── ...
    │	│   └── ...
    │	└── sampled					# Output Folder
    │	    ├── json 						# Json file after running random_sample.py
    │	    │	├── channel 1						# Name of the channel 
    │	    │	│   ├── json file 1                				# Json file (file name = video_id) after sampling 
    │	    │	│   ├── json file 2						# Json file (file name = video_id) after sampling
    │	    │	│   └── ...
    │	    │   └── ...   
    │	    └── txt 						# Text file after running random_sample.py
    │	        ├── channel 1
    │	        │    ├── video_id					# Folder named after video_id
    │		│    │	 ├── text file                				# text file that contains only comments 
    │		│    │	 ├── lematizer_txt.txt					# contains both text and POS tag 
    │		│    │   ├── text_only.txt					# text file (comments after stemming)
    │        	│    │	 ├── tag_only.txt					# text file (POS tag for text_only.txt)
    │		│    │	 └── ...
    │		│    └── ...		
    │		└── ...       			
    ├── LICENSE          				
    └── README.md  


## Results
### Task 1: Sequence Labelling
We train sequence labelling model to identify target terms and aspect terms in every sentence. The dataset for this task is in CoNLL format.

| Model      | P      | R      | F1     |
|------------|--------|--------|--------|
| BiLSTM+CRF | 0.6070 | 0.5395 | 0.5707 |
| BERT       | 0.5814 | 0.5788 | 0.5798 |


### Task 2: Sentiment Classification
We train sentiment classification model to identify the sentiment polarity of a given aspect term and category. The dataset is in CSV format.

| Model  | Acc   | P     | R     | F1    |
|--------|-------|-------|-------|-------|
| BERT   | 0.800 | 0.804 | 0.800 | 0.799 |
| BiLSTM | 0.815 | 0.816 | 0.816 | 0.816 |
| CNN    | 0.811 | 0.812 | 0.811 | 0.811 |
| SVM    | 0.714 | 0.716 | 0.714 | 0.712 |
