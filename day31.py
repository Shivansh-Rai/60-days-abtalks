from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(api_key="", model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("Explain the following in simple terms:\n{input}")

chain = prompt | llm


print("Enter text:")
t = input()

res = chain.invoke({"input": t})

print("\nResponse:")
print(res.content)