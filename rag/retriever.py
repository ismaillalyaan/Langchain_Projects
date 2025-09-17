from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import tempfile, shutil, atexit


st.title("🩺 Cancer Information Chatbot")


@st.cache_resource
def load_faiss():
    loader = PyPDFLoader("Understanding-Cancer-What-is-Cancer-and-Types-of-Cancer_rev.pdf")
    text = loader.load()

    text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_split.split_documents(text)

    # Temporary FAISS index
    temp_dir = tempfile.mkdtemp()

    # Cleanup on exit
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    db = FAISS.from_documents(documents, OllamaEmbeddings(model="nomic-embed-text"))
    db.save_local(temp_dir)

    return db


db = load_faiss()
retriever = db.as_retriever()

llm = ChatOllama(model="llama3.2")

prompt = PromptTemplate(
    template=(
        "Answer the following question based ONLY on the provided context.\n"
        "Do NOT reveal chain-of-thought; give a concise, factual answer.\n\n"
        "<context>\n{context}\n</context>\n\n"
        "Question: {input}\n"
    ),
    input_variables=["context", "input"],
)

documents_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, documents_chain)


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if user_input := st.chat_input("Ask about cancer..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_input})
            answer = response.get("answer", "Sorry, I couldn’t find an answer.")
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
