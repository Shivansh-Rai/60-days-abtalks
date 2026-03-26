import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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


def process(t):
    t = clean(t)
    w = word_tokenize(t)
    st = set(stopwords.words('english'))
    w = [i for i in w if i not in st]
    p = PorterStemmer()
    w = [p.stem(i) for i in w]
    return w


def stats(w):
    total = len(w)
    unique = len(set(w))
    return np.array([total, unique])


def quiz():
    q = [
        ("What does TF in TF-IDF stand for?", "term frequency"),
        ("What is the purpose of stopword removal?", "remove common words"),
        ("What does stemming do?", "reduce words")
    ]
    score = 0
    for i in q:
        print(i[0])
        a = input().lower()
        if i[1] in a:
            score += 1
    return score


print("Enter text:")
t = input()

w = process(t)
s = stats(w)

print("Processed Words:", w)
print("Total Words:", s[0])
print("Unique Words:", s[1])

print("Quiz Time")
sc = quiz()
print("Score:", sc, "/3")