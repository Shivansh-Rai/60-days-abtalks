from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def build():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression())
    ])


print("Enter number of samples:")
n = int(input())

docs = []
y = []

print("Enter text and label (1=positive, 0=negative):")
for _ in range(n):
    docs.append(input())
    y.append(int(input()))

p = build()
p.fit(docs, y)

pred = p.predict(docs)
acc = accuracy_score(y, pred)

print("\nAccuracy:", acc)

print("\nEnter test text:")
t = input()

print("Predicted Sentiment:", p.predict([t])[0])