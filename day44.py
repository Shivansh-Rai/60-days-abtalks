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


def normalize(x):
    return x / np.linalg.norm(x)


def evaluate(query, docs):
    qv = normalize(embed(query)).reshape(1, -1)
    dist, ind = index.search(qv, 2)
    return [docs[i] for i in ind[0]]


print("Enter number of documents:")
n = int(input())

docs = []
vecs = []

for _ in range(n):
    t = input()
    docs.append(t)
    vecs.append(normalize(embed(t)))

vecs = np.array(vecs)

d = vecs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(vecs)


print("Enter query:")
q = input()

res = evaluate(q, docs)

print("\nRetrieved Documents:")
for i in res:
    print(i)