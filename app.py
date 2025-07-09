import streamlit as st
import os
from backend.loader import load_and_chunk_pdf
from backend.vector_store import save_chunks_to_vectorstore
from backend.qa_engine import build_qa_chain
import tempfile

st.set_page_config(page_title="🧠 Legal Document Q&A", layout="wide")
st.title("📚 Legal Document Question Answering System (RAG with Cohere)")

with st.sidebar:
    st.subheader("Upload Legal PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        chunks = load_and_chunk_pdf(temp_path)
        save_chunks_to_vectorstore(chunks)
        st.success("✅ PDF processed and indexed.")

st.markdown("### Ask a question from the uploaded legal document")

# ✅ Updated condition here
if uploaded_file:
    qa_chain = build_qa_chain()
    question = st.text_input("Ask your question:")
    if question:
        with st.spinner("Thinking..."):
            result = qa_chain(question)
            st.markdown(f"**Answer:** {result['result']}")
            with st.expander("📄 Source Documents"):
                for doc in result['source_documents']:
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    st.markdown(doc.page_content[:500] + "...")
else:
    st.info("📂 Please upload a legal PDF from the sidebar to begin.")
