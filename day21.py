import numpy as np


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
    w = t.split()
    st = {"is","and","the","a","an","to","in","of","for","on","at","this","that"}
    w = [i for i in w if i not in st]
    return w


def freq(w):
    d = {}
    for i in w:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def vector(w, vocab):
    v = []
    for i in vocab:
        v.append(w.count(i))
    return np.array(v)


print("Enter first text:")
t1 = input()

print("Enter second text:")
t2 = input()

w1 = process(t1)
w2 = process(t2)

vocab = list(set(w1 + w2))

v1 = vector(w1, vocab)
v2 = vector(w2, vocab)

s = cosine(v1, v2)

print("\nProcessed Text 1:", w1)
print("Processed Text 2:", w2)

print("\nVocabulary:", vocab)

print("\nSimilarity Score:", s)