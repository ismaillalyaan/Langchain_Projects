from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st


prompt = ChatPromptTemplate(
    [
        ('system','You are a helpful assistant'),
        ('user','Question:{question}')
    ]
)

st.title('Welcome to the simple chatbot using llama3.2!')
input_text = st.text_input('Ask the model!')

model = Ollama(model='llama3.2')

parser = StrOutputParser()

chain = prompt|model|parser

if input_text:
    st.write(chain.invoke({'question':input_text}))