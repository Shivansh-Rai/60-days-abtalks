from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(api_key="YOUR_API_KEY", model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template(
    "Summarize the following document in 3-4 concise points:\n\n{input}"
)

chain = prompt | llm


print("Enter document text:")
t = input()

res = chain.invoke({"input": t})

print("\nSummary:")
print(res.content)