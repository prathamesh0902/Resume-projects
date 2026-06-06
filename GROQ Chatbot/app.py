import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# ---------------------------
# Setup
# ---------------------------
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Please set the GROQ_API_KEY environment variable.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Groq Chatbot", page_icon="⚡")
st.title("⚡ Static Prompting Chatbot (Groq API)")

# User input
input_data = st.text_input("💬 Hey, what's on your mind?")

# Model selection
model_name = st.selectbox(
    "Choose a Groq model:",
    ["llama-3.3-70b-versatile", "llama2-70b-4096", "gemma-7b-it"]
)

# When button is clicked
if st.button("Answer"):
    if not input_data.strip():
        st.warning("Please enter a question or prompt.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Send request to Groq API
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_data}
                    ],
                    max_tokens=300
                )
                answer = response.choices[0].message.content
                st.markdown(f"**Answer:**\n\n{answer}")
            except Exception as e:
                st.error(f"Error: {e}")
