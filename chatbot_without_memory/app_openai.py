from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

import os

from dotenv import load_dotenv

prompt = ChatPromptTemplate([
    ('system','You are a helpful assistant!'),
    ('user','Question{question}')
])

st.title('Openai chatbot for testing!')
input_text = st.text_input("Ask the model!")

model = ChatOpenAI(model = 'gpt-4o')

parser = StrOutputParser()

chain = prompt|model|parser

if input_text:
    st.write(chain.invoke({'question':input_text}))