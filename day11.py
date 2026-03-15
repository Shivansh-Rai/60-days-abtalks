from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def vectorize(a, b):
    v = TfidfVectorizer()
    x = v.fit_transform([a, b])
    return x


def similarity(x):
    s = cosine_similarity(x[0:1], x[1:2])
    return s[0][0]


print("Enter first document:")
a = input()

print("Enter second document:")
b = input()

x = vectorize(a, b)
s = similarity(x)

print("Cosine Similarity Score:", s)