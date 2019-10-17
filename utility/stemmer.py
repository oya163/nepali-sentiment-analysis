import sys
import snowballstemmer

nepali_stemmer = snowballstemmer.NepaliStemmer()

def stem(word):
    result = nepali_stemmer.stemWord(word)
    if result != word:
        suffix = word[len(result):]
        return(result, suffix)
    else:
        return(result)


def read_file(filename, out_file):
    with open(filename, 'r', encoding='utf-8') as f, open(out_file, 'w', encoding='utf-8') as o_f:
        reader = f.readlines()
        for row in reader:
            words = row.split()
            stemmed_res = []
            for each in words:
                result = stem(each)
                if type(result) == tuple:
                    if result[0] == '':
                        stemmed_res.append(result[1])
                    else:
                        stemmed_res.append(result[0])
                        stemmed_res.append(result[1])
                else:
                    stemmed_res.append(result)
            print(row)
            print(stemmed_res)
            o_f.write(' '.join(stemmed_res)+'\n')
            

def main():
    sentence = ["नेपालीको", "नेपालमा",  "राम्रो",  "छ"]
    for each in sentence:
        print(stem(each))
    #read_file(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
