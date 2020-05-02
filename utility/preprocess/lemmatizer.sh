#!/bin/bash

lemma=".lemma"
find data/youtube/sampled/txt -name "*.txt" | xargs morfessor-segment -L data/morpheme/morpheme.sgm --output-format '{analysis} ' > xargs{$lemma}
