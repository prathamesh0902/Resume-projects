import streamlit as st
from langchain_community.llms import Ollama

st.title("Static Prompting Chatbot")
input_data = st.text_input("Hey Whats in your mind?")
model = Ollama(model = 'gemma3')
response = model.invoke(input_data)
if st.button("Answer"):
    st.write(response)