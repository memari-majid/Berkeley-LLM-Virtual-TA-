# -*- coding: utf-8 -*-
"""KnowledgeBase_RAG_with_PGVector.py"""

import os
import glob
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.llms import Ollama
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Function to load all PDF documents from a directory
def load_documents_from_directory(directory_path):
    '''
    Load all PDF documents from the specified directory.
    '''
    pdf_files = glob.glob(os.path.join(directory_path, '*.pdf'))
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        documents.extend(docs)
        print(f"Loaded {len(docs)} pages from {pdf_file}")
    return documents

# Function to split loaded documents into smaller chunks for processing
def split_documents(documents):
    '''
    Split documents into smaller chunks for better processing with the LLM.
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")
    return split_docs

# Function to vectorize documents using PGVector and HuggingFace embeddings
def create_vector_store(documents, collection_name, connection_string):
    '''
    Vectorize documents for similarity search using PGVector and HuggingFace embeddings.
    '''
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
    )
    return vector_store

# Function to scrape web page content and convert it into LangChain documents
def scrape_web_page(url):
    '''
    Scrapes the text content from a web page and converts it into a LangChain Document.
    '''
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        document = Document(page_content=text)
        return [document]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the web page: {e}")
        return []

# Create the RAG Chain
def create_rag_chain(vector_store, prompt, llm):
    '''
    Create a RAG chain by connecting the retriever, LLM, and output parser.
    '''
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main():
    # Directory containing the PDF papers
    directory_path = "./papers"

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    # Load and process all PDF documents in the directory
    documents = load_documents_from_directory(directory_path)
    if not documents:
        print("No documents found in the directory.")
        return

    split_docs = split_documents(documents)

    # Define connection parameters
    collection_name = "papers_collection"  # Replace with your desired collection name
    connection_string = "postgresql://username:password@hostname:port/database"  # Replace with your connection string

    # Create or connect to the vector store
    vector_store = create_vector_store(split_docs, collection_name, connection_string)

    # Optionally, scrape a web page and add the content to the RAG system
    url = input("Enter a URL to scrape and add to the knowledge base (or press Enter to skip): ")
    if url:
        scraped_documents = scrape_web_page(url)
        if scraped_documents:
            split_scraped_docs = split_documents(scraped_documents)
            vector_store.add_documents(split_scraped_docs)
            print("Scraped documents added to the vector store.")

    # Create the prompt template for LLaMA model interaction
    prompt_template = PromptTemplate(
        template="Answer the following question based on the context: {context}. Question: {question}",
        input_variables=["context", "question"]
    )

    # Load the local LLaMA model (Ollama)
    llm = Ollama(model="llama3.2")

    # Create the RAG chain
    rag_chain = create_rag_chain(vector_store, prompt_template, llm)

    # Interactive question loop
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        response = rag_chain.invoke(question)
        print("\nRAG Chain Response:")
        print(response)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
