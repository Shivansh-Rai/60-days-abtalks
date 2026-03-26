from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


def train(docs, y):
    v = TfidfVectorizer()
    x = v.fit_transform(docs)
    m = LogisticRegression()
    m.fit(x, y)
    p = m.predict(x)
    return v, m, p


def evaluate(y, p):
    pr = precision_score(y, p)
    re = recall_score(y, p)
    f1 = f1_score(y, p)
    return pr, re, f1


print("Enter number of samples:")
n = int(input())

docs = []
y = []

print("Enter text and label:")
for _ in range(n):
    docs.append(input())
    y.append(int(input()))

v, m, p = train(docs, y)

pr, re, f1 = evaluate(y, p)

print("\nPrecision:", pr)
print("Recall:", re)
print("F1 Score:", f1)

print("\nEnter test text:")
t = input()

x = v.transform([t])
print("Predicted Label:", m.predict(x)[0])