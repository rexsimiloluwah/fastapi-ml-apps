import nltk
import numpy as np 
from typing import List
from nltk.stem.porter import PorterStemmer 
porter_stemmer = PorterStemmer()

def tokenize_sentence(sentence:str)->List[str]:
    return nltk.word_tokenize(sentence)

def stem(word:str)->str:
    return porter_stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence:List[str],all_words:List[str])->np.ndarray:
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx,word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] += 1.0
    return bag 

if __name__ == "__main__":
    sentence = "Hello, I am glad to see you."
    print(tokenize_sentence(sentence))
    stem_words = ["Organize","organization","organs"]
    print([stem(word) for word in stem_words])

    print(bag_of_words(tokenize_sentence("I was seeing you yesterday"), sentence.lower().split(" ")))