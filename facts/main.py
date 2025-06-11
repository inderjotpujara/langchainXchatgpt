from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
# This script loads a text file, splits it into chunks, and prepares it for further processing with LangChain.
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0, # Adjust chunk size and overlap as needed  
  )

loader = TextLoader("./facts.txt")

docs = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="emb"
)

results = db.similarity_search("What is an interesting fact about english language?")

for result in results:
    print("\n")
    # print(f"Score: {score}")
    print(result.page_content)
# The above code loads a text file, splits it into chunks, and creates a vector store using Chroma.
# The documents are now ready for use in similarity search or other LangChain operations.







# commenting the below lines to view the implementation of document loading later 
# print("Loaded documents:")
# for doc in document:
#     print(doc.page_content)
#     print("\n")
# print(f"Total documents loaded: {len(documents)}") 
# print(f"doc loaded", document)

# You can now use these documents with LangChain or any other processing you need.
# For example, you can create a vector store or use them in a chain.