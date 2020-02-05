'''
    Create word vectors using Gensim
	
	Author - Oyesh Mann Singh
	Date - 12/31/2019
	
	How to run:
		python word_vectors.py
'''

import os
import argparse
from gensim.models import Word2Vec, KeyedVectors, FastText

parser = argparse.ArgumentParser(add_help=True, description=('Text Collector Parser'))
parser.add_argument('--input_dir', '-idir', default='../data/channels/comments', metavar='PATH', help='Input path directory')
parser.add_argument('--output_dir', '-odir', default='../data/embeddings', metavar='PATH', help='Output path directory')
parser.add_argument('--embeddings', '-e', default='word2vec', choices=['word2vec', 'fasttext'], metavar='STR', help='Embeddings')
parser.add_argument('--embed_type', '-t', default=0, choices=[0, 1], metavar='INT', help='Embeddings Type 0: skip_gram, 1: cbow, default: 0}')
parser.add_argument('--eval_mode', '-m', default=False, action="store_true", help='Evaluation mode')
parser.add_argument('--similarity_check', '-s', default=['नेपाल', 'Nepal'], metavar='STR', help='Similarity check word')


args = parser.parse_args()


def main():
    input_dir = args.input_dir
    output_dir = args.output_dir
    embeddings = args.embeddings
    embed_type = args.embed_type
    similarity_check = args.similarity_check
    
    if embeddings == 'word2vec':
        output_dir = os.path.join(output_dir, 'nep2vec')
    else:
        output_dir = os.path.join(output_dir, 'nep2ft')
    
    output_file = os.path.join(output_dir, 'embeddings.vec')

    # Training mode
    if not args.eval_mode:
        print("Training {0} model".format(embeddings))
        sents = []
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                input_file = os.path.join(root, f)
                i_f = open(input_file, 'r', encoding='utf8')
                for line in i_f:
                    if len(line) > 0:
                        sents.append(line.split())

        if embeddings == 'word2vec':
            model = Word2Vec(sents, size=300, sg=embed_type, workers=10)
            
        elif embeddings == 'fasttext':            
            model = FastText(size=4, window=5, min_count=1)
            model.build_vocab(sentences=sents)
            total_examples = model.corpus_count
            model.train(sentences=sents, total_examples=total_examples, epochs=5)            
                
        model.wv.save_word2vec_format(output_file, binary=False)
    
    # Eval mode mode
    model = KeyedVectors.load_word2vec_format(output_file, binary=False)
    
    print("Checking word similarity from: ", embeddings)
    for every in similarity_check:
        print(model.most_similar(every, topn=10))
    
    # Print info
    print("Length of vocabulary ",model.wv.vectors.shape[0])
    
    # To-do
    # print how much data is code-mixed, native and romanized

if __name__ == "__main__":
	main()
