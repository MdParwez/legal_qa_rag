from langchain.chains import RetrievalQA
from langchain_community.llms import Cohere
from backend.vector_store import load_vectorstore
import os
from dotenv import load_dotenv

load_dotenv()

def build_qa_chain():
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = Cohere(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="command-r-plus",  # or "command-r", "command-xlarge-nightly", etc.
        temperature=0.3
    )

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain
