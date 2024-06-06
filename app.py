import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["cohere_api_key"] = os.getenv("COHERE_API_KEY")
os.environ["TAVILY_API_KEY"] =os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]='true'
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

llm=ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")

#RAG
def get_pdf_text(pdf_docs):
    text=""
    try:
        for pdf in pdf_docs:
            pdf_reader= PdfReader(pdf)
            for page in pdf_reader.pages:
                text+= page.extract_text()
    except Exception as e:
        print(e)
    return text

def get_text_chunks(docs):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    chunks=text_splitter.split_text(docs)
    return chunks

def get_vector_store(chunks):
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    vector_store=Chroma.from_texts(chunks,embeddings)
    retriever=vector_store.as_retriever()
    return retriever

def main():
    #Chatbot UI
    st.title("LearnAura")
    global tools
    tools=[]

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                retriever=get_vector_store(text_chunks) 
                print(retriever)
                #Retriever Tool
                doc_retriever = create_retriever_tool(retriever,"Document Retriever Tool","A versatile tool designed for efficiently locating and extracting relevant information from the documents provided by the user. It is ideal for gathering comprehensive insights and specific data from user-supplied documents.")
                tools=[doc_retriever]
                print(doc_retriever)
                st.success("Done")

    #Tavily Search
    search = TavilySearchAPIWrapper()
    tavily=TavilySearchResults(name="tavily",api_wrapper=search)
    tools.append(tavily)               

    #Wikipedia
    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
    tools.append(wiki)

    #Arxiv      
    arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
    tools.append(arxiv)

    prompt_agent = ChatPromptTemplate.from_messages([
    ("system", "You are Aura, a personal companion and your duty is to help students to study and excel in their academics. Use the provided documents to answer the questions."),
    ("system", "While answerng to any query, you should choose the tools wisely. Don't search everything in wikipedia or other tools. First try to answer the query with the information,you already possess. Then only you should go and use the different tools. Also mention the tool which you have used for getting the result"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")])

    agent = create_tool_calling_agent(llm, tools, prompt_agent)
    agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Capture the invoked response and tool usage
            try:
                response = agent_executor.invoke({
                            "input": prompt,
                            "chat-history": [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ]
                        })
                content_value=response['output']
                print(response)
            except Exception as e:
                print(e)
                content_value = "I'm sorry, but I couldn't generate a response."
            
            st.markdown(content_value)
        
        st.session_state.messages.append({"role": "assistant", "content": content_value})

        hello='''response = llm.invoke(
                input=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            )
            # Extract the 'content' value
            content_value = next((item[1] for item in response if item[0] == 'content'), None)
            '''
        
if __name__=="__main__":
    main()