import numpy as np
from scipy.stats import entropy


def clean(t):
    s = ""
    for c in t:
        if c.isalnum() or c.isspace():
            s += c
        else:
            s += " "
    return s.lower()


def freq(w):
    d = {}
    for i in w:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d


print("Enter text:")
t = input()

t = clean(t)
w = t.split()

d = freq(w)

words = list(d.keys())
counts = np.array(list(d.values()), dtype=float)

p = counts / counts.sum()

print("\nWords:")
print(words)

print("\nFrequencies:")
print(counts)

print("\nProbability Distribution:")
print(p)

print("\nEntropy of Distribution:")
print(entropy(p))