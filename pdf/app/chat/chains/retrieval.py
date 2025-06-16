from langchain.chains import ConversationalRetrievalChain
from app.chat.chains.streamable import StreamableChain

class StreamingConversationalRetrievalChain(StreamableChain, ConversationalRetrievalChain):
    """
    A streaming version of the ConversationalRetrievalChain.
    This chain allows for streaming responses from the LLM while retrieving relevant documents.
    """
    pass