import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def clean(t):
    s = ""
    for c in t:
        if c.isalnum() or c.isspace():
            s += c
        else:
            s += " "
    return s.lower()


def tokenize(t):
    return t.split()


def pipeline(t):
    t = clean(t)
    w = tokenize(t)
    st = {"is","and","the","a","an","to","in","of","for","on","at","this","that"}
    w = [i for i in w if i not in st]
    return w


def evaluate(docs):
    p = [" ".join(pipeline(d)) for d in docs]
    v = TfidfVectorizer()
    x = v.fit_transform(p)
    total = sum(len(pipeline(d)) for d in docs)
    unique = len(v.get_feature_names_out())
    return p, total, unique


print("Enter number of documents:")
n = int(input())

docs = []
for _ in range(n):
    docs.append(input())

p, total, unique = evaluate(docs)

print("\nProcessed Documents:")
print(p)

print("\nTotal Tokens:", total)
print("Unique Vocabulary Size:", unique)