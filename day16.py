import numpy as np


def dot(a, b):
    return np.dot(a, b)


def matmul(a, b):
    return np.matmul(a, b)


print("Enter vector dimension:")
d = int(input())

print("Enter first vector:")
v1 = np.array(list(map(float, input().split())))

print("Enter second vector:")
v2 = np.array(list(map(float, input().split())))

print("\nDot Product:")
print(dot(v1, v2))

print("\nEnter rows and columns of matrix A:")
r1, c1 = map(int, input().split())

print("Enter matrix A:")
a = []
for _ in range(r1):
    a.append(list(map(float, input().split())))
a = np.array(a)

print("\nEnter rows and columns of matrix B:")
r2, c2 = map(int, input().split())

print("Enter matrix B:")
b = []
for _ in range(r2):
    b.append(list(map(float, input().split())))
b = np.array(b)

print("\nMatrix Multiplication Result:")
print(matmul(a, b))

