# Nepali Sentiment Analysis

With an increase in internet access in Nepal and ease of typing in Nepali, Nepali language has become prevalent on social media platforms. However, there is a lack of fine-grained sentiment analysis for comments written in Nepali. In this project, we aim to analyze the YouTube comments written in Nepali, both code-mixed and code-switched. As a first step, we are creating a Named Targeted Aspect Based Sentiment Analysis Dataset from comments extracted from popular Nepali YouTube videos under the News & Politics category. This dataset can be used for empirical studies on multilingual training, domain adaptation and vector-space word mapping techniques. Hence, this dataset can play a major role in identifying abusive comments in Nepali texts.

## Data Collection
Nepali Sentiment Analysis (NepSA) is a named targeted aspect-based sentiment analysis dataset. We collected the comments from the most popular Nepali YouTube channels having the highest subscribers under the News & Politics category. The dataset consists of 3068 comments extracted from  37 different YouTube videos of 9 different YouTube channels. We used binary sentiment polarity schema and divided the comments into 6 aspect categories General, Profanity, Violence, Feedback, Sarcasm and Out-of-scope to annotate the data. All the targeted annotations are created considering the target entity towards which the sentiment is expressed and not on the general understanding of the sentence. The target entities are divided mainly into Person, Organization, Location and Miscellaneous.

## Tasks
We divided the experiments into two subtasks: Aspect Term Extraction and Sentiment Polarity Identification. 

Aspect Term Extraction resembles the sequence labelling task where we tag each token of a given sentence with predefined aspect category or named entities. We experiment with four major categories General, Profanity, Violence, Feedback under Aspect Category and Person, Organization, Location and Miscellaneous under Target Entities. 

Sentiment Polarity Identification is a binary classification task to identify sentiment polarity [0, 1] of each aspect categories in every given sentence.

## How to run

### Task 1: Aspect Term Extraction

Please refer modified version of torchnlp [here](https://github.com/oya163/torchnlp)
    

### Task 2: Sentiment Polarity Identification

    time bash run_classification.sh    


## Results
### Task 1: Aspect Term Extraction
| Model      | P      | R      | F1     |
|------------|--------|--------|--------|
| BiLSTM+CRF | 0.6070 | 0.5395 | 0.5707 |
| BERT       | 0.5814 | 0.5788 | 0.5798 |


### Task 2: Sentiment Polarity Identification
| Model  | Acc   | P     | R     | F1    |
|--------|-------|-------|-------|-------|
| BERT   | 0.800 | 0.804 | 0.800 | 0.799 |
| BiLSTM | 0.815 | 0.816 | 0.816 | 0.816 |
| CNN    | 0.811 | 0.812 | 0.811 | 0.811 |
| SVM    | 0.714 | 0.716 | 0.714 | 0.712 |



## References
- [Internet Stat](https://www.Internetworldstats.com/stats3.htm#asia)
- [They Don't Leave Us Alone Anywhere We Go](https://research.google/pubs/pub47721/)
- SemEval [2014](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools), [2015](http://alt.qcri.org/semeval2015/task12/) and [2016](http://alt.qcri.org/semeval2016/task5/) Aspect Based Sentiment Analysis
- [Annotating Targets of Opinions in Arabic using Crowdsourcing](https://www.aclweb.org/anthology/W15-3210.pdf)



## Publication
- Published in [IEEE ASONAM 2020](https://ieeexplore.ieee.org/document/9381292)

## Citation
    @INPROCEEDINGS{9381292,
      author={Singh, Oyesh Mann and Timilsina, Sandesh and Bal, Bal Krishna and Joshi, Anupam},
      booktitle={2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)}, 
      title={Aspect Based Abusive Sentiment Detection in Nepali Social Media Texts}, 
      year={2020},
      volume={},
      number={},
      pages={301-308},
      doi={10.1109/ASONAM49781.2020.9381292}
    }
