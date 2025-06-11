from langchain.vectorstores.chroma import Chroma;
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv


load_dotenv()

chat  = ChatOpenAI();

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = RedundantFilterRetriever(
    chroma=db,
    embeddings=embeddings
)

# retrieval chain below
retriever = db.as_retriever()
chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever,
    # return_source_documents=True
)

result = chain.run("What is an interesting fact about english language?")
print("\n")
print(result)