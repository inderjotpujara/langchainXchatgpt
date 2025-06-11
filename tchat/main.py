from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose = True)

memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True,
    llm=chat
)

prompt = ChatPromptTemplate(
  input_variables=["content", "messages"],
  messages=[
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}")
  ]
)

chain = LLMChain(
  llm=chat,
  prompt=prompt,
  memory=memory,
  verbose=True
)



while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    result = chain({
        "content": user_input
    })
    response = result["text"]
    print("ChatGPT:", response)