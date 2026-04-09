from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory


llm = ChatOpenAI(api_key="YOUR_API_KEY", model="gpt-4o-mini")

memory = ConversationBufferMemory(return_messages=True)


print("Chatbot started (type 'exit' to stop)")

while True:
    q = input("You: ")
    
    if q.lower() == "exit":
        break
    
    memory.chat_memory.add_user_message(q)
    
    r = llm.invoke(memory.chat_memory.messages)
    
    memory.chat_memory.add_ai_message(r.content)
    
    print("Bot:", r.content)