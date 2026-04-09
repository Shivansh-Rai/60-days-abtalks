from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


llm = ChatOpenAI(api_key="YOUR_API_KEY", model="gpt-4o-mini")

history = []

print("Chatbot started (type 'exit' to stop)")

while True:
    q = input("You: ")
    
    if q.lower() == "exit":
        break
    
    history.append(HumanMessage(content=q))
    
    r = llm.invoke(history)
    
    history.append(AIMessage(content=r.content))
    
    print("Bot:", r.content)