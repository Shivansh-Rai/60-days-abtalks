import numpy as np


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def distance(a, b):
    return np.linalg.norm(a - b)


def best_pair(x):
    n = len(x)
    best = -1
    pair = (0, 0)
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine(x[i], x[j])
            if s > best:
                best = s
                pair = (i, j)
    return pair, best


print("Enter dimension:")
d = int(input())

print("Enter number of vectors:")
n = int(input())

x = []
for _ in range(n):
    v = list(map(float, input().split()))
    x.append(v)

x = np.array(x)

print("\nCosine Similarities:")
for i in range(n):
    for j in range(i + 1, n):
        print(i, j, cosine(x[i], x[j]))

print("\nEuclidean Distances:")
for i in range(n):
    for j in range(i + 1, n):
        print(i, j, distance(x[i], x[j]))

p, s = best_pair(x)

print("\nMost Similar Pair (Cosine):")
print(p, s)