from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def build():
    p = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression())
    ])
    return p


print("Enter number of samples:")
n = int(input())

docs = []
y = []

print("Enter text and label:")
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

print("Prediction:", p.predict([t])[0])

# 6
# Win free money now
# 1
# Meeting tomorrow at office
# 0
# Claim your free prize
# 1
# Project discussion today
# 0
# Exclusive offer just for you
# 1
# Team meeting schedule
# 0
# Free reward waiting for you