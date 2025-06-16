from langchain.chat_models import ChatOpenAI
from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from app.chat.models import ChatArgs
from app.chat.vector_stores.pinecone import build_retriever
from app.chat.llms.chatopenai import build_llm
from app.chat.memories.sql_memory import build_memory

def build_chat(chat_args: ChatArgs):
    retriever = build_retriever(chat_args)
    condense_question_llm = ChatOpenAI(streaming=False)
    llm = build_llm(chat_args)
    memory = build_memory(chat_args)

    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        condense_question_llm=condense_question_llm,
    )