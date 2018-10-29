import re
import pickle
from sklearn.externals import joblib
import os
from collections import Counter
from sklearn.svm import SVC
import numpy as np

word = re.compile('[\w]+')

stopwords = ['which', 'why', 'there', 'what', 'where', 'how', 'to', 'which', 'a', 'the', 'all', 'in', 'out', 'of', 'is', 'but', 'it', 'that', 'have', 'be', 'for', 'are', 'as']

model_folder = 'models'

class Classifier:
    def __init__(self):
        with open(os.path.join(model_folder, 'classifier.model'), 'rb') as cls_file:
            self.cls = joblib.load(cls_file)
        with open(os.path.join(model_folder, 'word2id'), 'rb') as id_file:
            self.word2id = pickle.load(id_file)

    def tokenize(self, text):
        parts = text.lower().strip().split()
        words = []
        for p in parts:
            w = re.findall(word, p)
            words.extend(w)
        return words   

    def vectorize(self, tokens):
        ids = [self.word2id[t] for t in tokens if t in self.word2id]
        counts = Counter(ids)
        one_hot = [0]*self.cls.shape_fit_[1]
        for i in counts:
            one_hot[i] = counts[i]
        return one_hot

    def predict(self, text):
        X = np.vstack([self.vectorize(self.tokenize(text))])
        result = self.cls.predict(X)
        return result[0]  

if __name__ == '__main__':
    book_q = 'What polyol do with productive rock?'
    tables_q = 'What is the total feedstock for Sulphur recovery for benchmark plant (in kmol/h)?'
    lists_q = 'How to do planned sidetrack off cement plug in AutoTrak systems?'
    classifier = Classifier()
    print('Book example: ', classifier.predict(book_q))
    print('Tables example: ', classifier.predict(tables_q))
    print('Lists example: ', classifier.predict(lists_q))
