import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma



def subtile_pdf_to_db(api_key, directory_path=r'.\data'):
    # 1. Load all PDFs from the directory, where name starts with "subtitle"
    all_docs = []
    for file in os.listdir(directory_path):
        if file.endswith(".pdf") and file.startswith("subtitle_"):
            file_path = os.path.join(directory_path, file)
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())
    
    if not all_docs:
        print("No PDF files found in the directory.")
    
    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # This the the splits variable
    splits = text_splitter.split_documents(all_docs)
    
    # 3. Create Embedding and Store in ChromaDB
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(api_key=api_key),
        persist_directory="./chroma_db"
    )

    return vector_db

def subtile_md_to_db(api_key, directory_path=r'.\data'):
    # 1. Load all Markdowns from the directory, where name starts with "subtitle"
    all_docs = []
    for file in os.listdir(directory_path):
        if file.endswith(".md") and file.startswith("subtitle_"):
            file_path = os.path.join(directory_path, file)
            loader = TextLoader(file_path, encoding='utf-8')
            all_docs.extend(loader.load())
    
    if not all_docs:
        print("No Markdown files found in the directory.")
    
    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # This the the splits variable
    splits = text_splitter.split_documents(all_docs)
    
    # 3. Create Embedding and Store in ChromaDB
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(api_key=api_key),
        persist_directory="./chroma_db"
    )

    return vector_db