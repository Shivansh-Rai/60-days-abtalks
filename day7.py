import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')


def clean(t):
    s = ""
    for c in t:
        if c.isalnum() or c.isspace():
            s += c
        else:
            s += " "
    return s.lower()


def tokenize(t):
    return word_tokenize(t)


def remove_stop(w):
    st = set(stopwords.words('english'))
    return [i for i in w if i not in st]


def stem(w):
    p = PorterStemmer()
    return [p.stem(i) for i in w]


def pipeline(t):
    t = clean(t)
    w = tokenize(t)
    w = remove_stop(w)
    w = stem(w)
    return w


def stats(w):
    l = len(w)
    u = len(set(w))
    return np.array([l, u])


if __name__ == "__main__":
    print("Enter text:")
    t = input()

    w = pipeline(t)
    s = stats(w)

    print("Processed Words:", w)
    print("Total Words:", s[0])
    print("Unique Words:", s[1])