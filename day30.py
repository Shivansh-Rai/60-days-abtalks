from openai import OpenAI

client = OpenAI(api_key="############")


def summarize(t):
    p = "Summarize the following text in 3 concise points:\n" + t
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": p}]
    )
    return r.choices[0].message.content


def qa(t, q):
    p = "Context:\n" + t + "\n\nQuestion:\n" + q + "\nAnswer clearly:"
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": p}]
    )
    return r.choices[0].message.content


print("Enter text:")
t = input()

print("\nSummary:")
print(summarize(t))

print("\nEnter question:")
q = input()

print("\nAnswer:")
print(qa(t, q))