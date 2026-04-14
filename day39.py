import numpy as np
import faiss
from openai import OpenAI


client = OpenAI(api_key="YOUR_API_KEY")


def embed(t):
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=t
    )
    return np.array(r.data[0].embedding, dtype="float32")


print("Enter number of documents:")
n = int(input())

docs = []
vecs = []

for _ in range(n):
    t = input()
    docs.append(t)
    vecs.append(embed(t))

vecs = np.array(vecs)

d = vecs.shape[1]

index = faiss.IndexFlatL2(d)
index.add(vecs)


print("Enter query:")
q = input()

print("Enter k (top results):")
k = int(input())

qv = embed(q).reshape(1, -1)

dist, ind = index.search(qv, k)

print("\nTop Matching Documents:")
for i in ind[0]:
    print(docs[i])