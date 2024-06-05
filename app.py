import streamlit as st
import os
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]='true'
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

st.title("LearnAura")

llm=ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

#Chatbot UI
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a virtual assistant that helps students to study and excel in their academics"}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = llm.invoke(
            input=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        )
        # Extract the 'content' value
        content_value = next((item[1] for item in response if item[0] == 'content'), None)
        st.markdown(content_value)
    st.session_state.messages.append({"role": "assistant", "content": content_value})