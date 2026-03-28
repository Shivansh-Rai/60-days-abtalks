from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train(docs, y):
    v = TfidfVectorizer()
    x = v.fit_transform(docs)

    m = LogisticRegression()

    p = {
        "C": [0.1, 1, 10],
        "max_iter": [100, 200]
    }

    g = GridSearchCV(m, p, cv=3)
    g.fit(x, y)

    return v, g


print("Enter number of samples:")
n = int(input())

docs = []
y = []

print("Enter text and label:")
for _ in range(n):
    docs.append(input())
    y.append(int(input()))

v, g = train(docs, y)

print("\nBest Parameters:")
print(g.best_params_)

print("Best Score:", g.best_score_)

print("\nEnter test text:")
t = input()

x = v.transform([t])
print("Predicted Label:", g.predict(x)[0])


# input cases 

# 6
# Win free money now
# 1
# Meeting tomorrow at office
# 0
# Claim your free prize
# 1
# Project discussion scheduled
# 0
# Exclusive offer just for you
# 1
# Team meeting today
# 0
# Free cash reward available