import numpy as np
import matplotlib.pyplot as plt


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


def freq(w):
    d = {}
    for i in w:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d


def top_k(d, k):
    items = list(d.items())
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:k]
    words = [i[0] for i in items]
    counts = np.array([i[1] for i in items])
    return words, counts


def plot(words, counts):
    plt.figure()
    plt.bar(words, counts)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


print("Enter text:")
t = input()

t = clean(t)
w = tokenize(t)
d = freq(w)
words, counts = top_k(d, 20)

print("Top Words:", words)
print("Counts:", counts)

plot(words, counts)