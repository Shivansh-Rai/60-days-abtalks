from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train(docs, y):
    v = TfidfVectorizer()
    x = v.fit_transform(docs)
    m = LogisticRegression()
    m.fit(x, y)
    p = m.predict(x)
    acc = accuracy_score(y, p)
    return v, m, acc


def predict(v, m, t):
    x = v.transform([t])
    return m.predict(x)[0]


print("Enter number of samples:")
n = int(input())

docs = []
y = []

print("Enter text and label:")
for _ in range(n):
    docs.append(input())
    y.append(int(input()))

v, m, acc = train(docs, y)

print("\nTraining Accuracy:", acc)

print("Enter test text:")
t = input()

p = predict(v, m, t)

print("Predicted Label:", p)