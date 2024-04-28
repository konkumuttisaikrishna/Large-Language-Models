"""
Question Answering with Retrieval QA and LangChain Language Models featuring FAISS vector stores.
This script uses the LangChain Language Model API to answer questions using Retrieval QA 
and FAISS vector stores. It also uses the Mistral huggingface inference endpoint to 
generate responses.
"""
import gradio as gr
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.llms import HuggingFaceHub


from pathlib import Path
from unidecode import unidecode

# set this key as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets['huggingface_token']


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    model = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def conversationalchain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        #repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
        #repo_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
        model_kwargs={"temperature": 0.9, "max_length": 2048},
    )
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversationalchain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversationalchain

def handle_userinput(user_question:str):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i,message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            text2=message.content 
            st.write(text2)
        else:
            text1=message.content 
            st.write(text1)

def main():
    st.set_page_config(
        page_title="Chat Bot PDFs",
        page_icon=":books:",
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.header("Chat Bot PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if st.button("Answer"):
        with st.spinner("Answering"):
            handle_userinput(user_question)
    if st.button("clear"):
        with st.spinner("Clearing"):
            st.cache_data.clear()

    with st.sidebar:
        st.header("your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("process"):
            with st.spinner("processing"):

                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = conversationalchain(vectorstore)
                
                st.write("compelete build model")


if __name__ == "__main__":
    main()  
