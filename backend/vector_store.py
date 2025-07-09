import os
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "embed-english-v3.0"  # or "embed-multilingual-v3.0"

def save_chunks_to_vectorstore(chunks, persist_dir="db"):
    embedding = CohereEmbeddings(
        model=EMBED_MODEL,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_dir)

def load_vectorstore(persist_dir="db"):
    embedding = CohereEmbeddings(
        model=EMBED_MODEL,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    return vectordb
