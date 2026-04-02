from openai import OpenAI

client = OpenAI(api_key=" removed api key ")

def generate(p):
    r = client.chat.completions.create(
        model="gpt5",
        messages=[{"role": "user", "content": p}]
    )
    return r.choices[0].message.content


print("Enter prompt:")
p = input()

res = generate(p)


print("\nResponse:")
print(res)