import numpy as np


def norm(v):
    return v / np.linalg.norm(v)


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("Enter dimension:")
d = int(input())

print("Enter first vector:")
v1 = np.array(list(map(float, input().split())))

print("Enter second vector:")
v2 = np.array(list(map(float, input().split())))

s1 = cosine(v1, v2)

n1 = norm(v1)
n2 = norm(v2)

s2 = cosine(n1, n2)

print("\nSimilarity before normalization:")
print(s1)

print("\nNormalized vectors:")
print(n1)
print(n2)

print("\nSimilarity after normalization:")
print(s2)