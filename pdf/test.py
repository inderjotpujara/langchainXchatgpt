from threading import Thread
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from queue import Queue

load_dotenv()


class StreamingHandler(BaseCallbackHandler):
  def __init__(self, queue: Queue):
    self.queue = queue
    
  def on_llm_new_token(self, token: str, **kwargs):
    # print(f"New token: {token}")
    self.queue.put(token)
    
  def on_llm_end(self, response, **kwargs):
    # print(f"LLM response: {response}")
    self.queue.put(None)
    
  def on_llm_error(self, error: Exception, **kwargs):
    print(f"Error: {error}")
    self.queue.put(None)

chat = ChatOpenAI(
    streaming=True,
    temperature=0.0
)



# chat = ChatOpenAI(streaming=True)

prompt = ChatPromptTemplate.from_messages([
  ("human", "{content}")
])

class StreamableChain:
  def stream(self, input: str):
    queue = Queue()
    handler = StreamingHandler(queue)
    def task():
      self(input, callbacks=[handler])
      
    Thread(target=task).start()
    
    while True:
      token = queue.get()
      if token is None:
        break
      yield token

class StreamingChain(StreamableChain, LLMChain):
    pass

chain = StreamingChain(llm=chat, prompt=prompt)



for output in chain.stream(input={"content": "tell me a joke"}):
    print(output)

print(chain("tell me a joke"))

# chain = LLMChain(
#     llm=chat,
#     prompt=prompt
# )

# output = chain("tell me a joke") 
# print(output)
# # messages = prompt.format_messages(content="tell me a joke")




# for message in chat.stream(messages):
#   print(message.content)
 