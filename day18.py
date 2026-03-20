import numpy as np
from scipy.optimize import minimize


def cost(x, t):
    return -np.dot(x, t) / (np.linalg.norm(x) * np.linalg.norm(t))


print("Enter dimension:")
d = int(input())

print("Enter target vector:")
t = np.array(list(map(float, input().split())))

print("Enter initial vector:")
x0 = np.array(list(map(float, input().split())))

res = minimize(cost, x0, args=(t,), method='BFGS')

print("\nOptimized Vector:")
print(res.x)

print("\nSimilarity Score:")
print(-cost(res.x, t))