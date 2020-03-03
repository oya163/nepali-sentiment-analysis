# Nepali Sentiment Analysis

With the increase in internet access in low resource Indic languages, these languages have become prevalent on social media platforms. Two of the most popular low resource Indic languages are Nepali and Bengali. However, there is a lack of fine-grained sentiment analysis tools for comments written in Nepali and Bengali. In this project, we aim to analyze the YouTube comments written in Nepali and Bengali, both code-mixed and code-switched. As a first step to developing a tool, we are creating a Named Targeted Aspect Based Sentiment Analysis Dataset from comments extracted from popular Nepali YouTube videos under the News & Politics category.  We intend to extend this work to create a similar dataset in Bengali. This dataset can be used for empirical studies on multilingual training, domain adaptation and vector-space word mapping techniques. The results will be a major component in systems for identifying abusive comments in South Asian and other low-resource languages.   

We are working on sentiment analysis on social media texts (Youtube comments) in Nepali language. As we can see from latest statistics, there is growing userbase in Facebook/Youtube compared to Twitter. We found out that lately lots of people are commenting in Youtube videos in Nepali language compared to Twitter. Plus, people are more likely to watch trending political development videos in Youtube compared to Twitter. Also, it is more convenient to scrap thousands of comments from a particular video in Youtube. Unfortunately, there is no any Nepali language analytics in Youtube comments. Therefore, our project focuses on sentiment analysis in Youtube comments written in Nepali language.


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
We train sequence labelling model to identify target term, target category, aspect term and aspect category in every comment. The dataset for this task is in CoNLL format

| Model                   | NepSA |
|-------------------------|-------|
| BiLSTM + CNN + CRF      | 53.33 |


### Task 2: Sentiment Classification
We train sentiment classification model to identify sentiment polarity of given aspect term and aspect category. We simply train this model in this format: Text + Aspect Category + Aspect Term -> Polarity

| Dataset    | Model | F1    |
|------------|-------|-------|
| Balanced   | LSTM  | 0.739 |
| Balanced   | CNN   | 0.767 |
| Unbalanced | LSTM  | 0.762 |
| Unbalanced | CNN   | 0.783 |
