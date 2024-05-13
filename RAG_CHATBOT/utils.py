from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.llms import HuggingFaceHub

import os
import tempfile
from typing import List
from tqdm import tqdm

def create_llm():
    """
    Creata an instance of mismistralai/Mistral-7B-Instruct-v0.2 format LLM using LlamaCpp

    returns:
    - llm: An instance mistralai/Mistral-7B-Instruct-v0.2
    """
    #create llm
    llm = HuggingFaceHub(
        repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.3, "max_length": 2048},
    )

    return llm

def create_vector_store(pdf_files:List):
    """
    Create In-memory FAISS vetor store using uploaded Pdf
    
    Args:
    - pdf_files(List): PDF file uploaded
    
    retunrs:
    - vector_store: In-memory Vector store fo further processing at chat app
    
    """
    vector_store = None
    
    if pdf_files:
        text = []
        
        for file in tqdm(pdf_files,desc="Processing files"):
            #Get the file and check it's extension
            file_extension = os.path.splitext(file.name)[1]
            
            #Write the PDF file to temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            #Load the PDF files using PyPdf library 
            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            #load if the text file:
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)
                
        #spilt the file into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=100)
        text_chunks = text_splitter.split_documents(text)
        
        #create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs={'device': 'cpu'})
        
        # Create vector store and storing document chunks using embedding model
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        
    return vector_store    

    

