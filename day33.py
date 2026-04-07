from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory #type: ignore
from langchain.chains import ConversationChain  #type: ignore 


llm = ChatOpenAI(api_key="YOUR_API_KEY", model="gpt-4o-mini")

memory = ConversationBufferMemory()

chat = ConversationChain(
    llm=llm,
    memory=memory
)


print("Chatbot started (type 'exit' to stop)")

while True:
    q = input("You: ")
    
    if q.lower() == "exit":
        break
    
    r = chat.predict(input=q)
    
    print("Bot:", r)