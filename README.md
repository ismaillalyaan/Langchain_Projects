# 🩺 Cancer Information Chatbot (LangChain + Streamlit)

This is a simple chatbot built with **LangChain** and **Streamlit** that answers questions about cancer based on a PDF file (`Understanding-Cancer-What-is-Cancer-and-Types-of-Cancer_rev.pdf`).  
The chatbot uses **local embeddings (Ollama)** and a **FAISS vector store** to perform retrieval-augmented generation (RAG).  

---

## 📌 Features
- Uploads and processes a **PDF file** with LangChain’s `PyPDFLoader`.
- Splits text into **chunks** using `RecursiveCharacterTextSplitter`.
- Generates **embeddings** locally via `OllamaEmbeddings` (model: `nomic-embed-text`).
- Stores embeddings in a **FAISS vector database**, cached for session speed.
- Automatically deletes temporary FAISS files when the app closes.
- Conversational UI built with **Streamlit Chat Components** (`st.chat_message`, `st.chat_input`).
- Maintains **chat history** with `st.session_state`.

---

## ⚙️ How It Works
1. **PDF Loader** → Extracts text from the cancer information PDF.  
2. **Text Splitter** → Splits content into manageable overlapping chunks.  
3. **Embeddings + Vector Store** → Encodes chunks into vectors (Ollama embeddings) and stores them in FAISS.  
4. **Retriever** → Finds the most relevant chunks for a user’s query.  
5. **LLM + Prompt** → Passes context + question into the `llama3.2` model for a precise answer.  
6. **Streamlit Chat UI** → Displays messages in a chatbot-like interface.

---
