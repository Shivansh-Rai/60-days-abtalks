from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType


llm = ChatOpenAI(api_key="YOUR_API_KEY", model="gpt-4o-mini")


@tool
def add(a: int, b: int) -> int:
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    return a * b


tools = [add, multiply]


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


print("Agent ready (type 'exit' to stop)")

while True:
    q = input("You: ")
    
    if q.lower() == "exit":
        break
    
    r = agent.run(q)
    
    print("Bot:", r)