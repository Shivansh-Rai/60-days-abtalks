import numpy as np


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def mean_vec(x):
    return np.mean(x, axis=0)


print("Enter embedding dimension:")
d = int(input())

print("Enter number of vectors:")
n = int(input())

x = []
for _ in range(n):
    v = list(map(float, input().split()))
    x.append(v)

x = np.array(x)

m = mean_vec(x)

print("\nMean Embedding:")
print(m)






if n >= 2:
    s = cosine(x[0], x[1])
    print("\nCosine Similarity between first two vectors:")
    print(s)