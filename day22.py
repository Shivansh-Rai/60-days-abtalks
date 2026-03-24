from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train(docs, y):
    v = TfidfVectorizer()
    x = v.fit_transform(docs)
    m = LogisticRegression()
    m.fit(x, y)
    return v, m


def predict(v, m, t):
    x = v.transform([t])
    return m.predict(x)[0]


print("Enter number of training samples:")
n = int(input())

docs = []
y = []

print("Enter text and label (0 or 1):")
for _ in range(n):
    t = input()
    l = int(input())
    docs.append(t)
    y.append(l)

v, m = train(docs, y)

print("Enter test text:")
t = input()

p = predict(v, m, t)

print("Predicted Label:", p)