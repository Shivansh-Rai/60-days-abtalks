from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def build():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression())
    ])


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
pr = precision_score(y, pred)
re = recall_score(y, pred)
f1 = f1_score(y, pred)

print("\nAccuracy:", acc)
print("Precision:", pr)
print("Recall:", re)
print("F1 Score:", f1)

print("\nEnter test text:")
t = input()

print("Prediction:", p.predict([t])[0])