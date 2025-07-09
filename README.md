# 🧠 Legal Document Q&A System (RAG with Cohere)

This project is a **Retrieval-Augmented Generation (RAG)** based **Legal Document Question Answering System** built using:

- 🦜 LangChain 
- 📄 PDF loaders
- ⚡ Cohere Embeddings & LLM
- 📚 Chroma vector store

---

## 📌 Features

- ✅ Upload legal PDF documents
- ✅ Chunk and embed text using Cohere's `embed-english-v3.0`
- ✅ Store document vectors in a local Chroma DB
- ✅ Ask natural language questions about the PDF
- ✅ Get AI-generated answers with source references
- ✅ Streamlit UI for easy interaction

---

## 📷 Demo

![Screenshot (998)](https://github.com/user-attachments/assets/ca920179-ae77-4b82-a5fa-7825d385c61b)

<!-- Add your own screenshot if needed -->

---

## 🏗️ Tech Stack

| Tool           | Purpose                          |
|----------------|----------------------------------|
| `LangChain`    | Orchestrate RAG pipeline         |
| `Cohere`       | Embeddings + LLM for Q&A         |
| `Chroma`       | Vector DB for semantic search    |
| `Streamlit`    | Web UI                           |
| `dotenv`       | Local secret handling            |
| `PyPDF2`       | PDF parsing                      |

---
