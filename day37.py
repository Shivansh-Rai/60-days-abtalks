from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")


def embed(t):
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=t
    )
    return r.data[0].embedding


print("Enter first document:")
t1 = input()

print("Enter second document:")
t2 = input()

e1 = embed(t1)
e2 = embed(t2)

print("\nEmbedding 1 (first 5 values):")
print(e1[:5])

print("\nEmbedding 2 (first 5 values):")
print(e2[:5])