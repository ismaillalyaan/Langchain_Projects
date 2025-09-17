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

st.title('Welcome to the simple chatbot using deepseek-r1!!')
input_text = st.text_input('Ask the model!')

model = Ollama(model='deepseek-r1:latest')

parser = StrOutputParser()

chain = prompt|model|parser

if input_text:
    with st.spinner("🤔 Thinking..."):
        raw_output = chain.invoke({'question': input_text})
    # Remove "thinking" if wrapped in <think> tags
    if "<think>" in raw_output and "</think>" in raw_output:
        answer = raw_output.split("</think>")[-1].strip()
    else:
        answer = raw_output
    st.write(answer)