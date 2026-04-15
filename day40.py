import numpy as np
from pymongo import MongoClient
from openai import OpenAI


client = OpenAI(api_key="###")

db_client = MongoClient("mongodb://localhost/")
db = db_client["ai_db"]
col = db["documents"]


def embed(t):
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=t
    )
    return r.data[0].embedding


print("Enter number of documents:")
n = int(input())

for _ in range(n):
    t = input()
    e = embed(t)
    
    doc = {
        "text": t,
        "embedding": e
    }
    
    col.insert_one(doc)

print("\nData stored in MongoDB")

print("\nStored Documents:")
for d in col.find():
    print(d["text"])