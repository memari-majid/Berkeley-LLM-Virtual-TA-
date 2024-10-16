# -*- coding: utf-8 -*-
"""KnowledgeBase_RAG_with_PGVector_and_WebUI.py"""

import os
import glob
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Imports from langchain and other libraries
from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document

import warnings

# Import Streamlit for the Web UI
import streamlit as st

# Import codecs and logging for text processing and logging
import codecs
import logging

# Import additional libraries for audio and image processing
from PIL import Image
import numpy as np
import torch

# Import OpenAI's Whisper for speech-to-text transcription
import whisper

# Import CLIP model for image embeddings
from transformers import CLIPProcessor, CLIPModel

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, filename='document_processing.log')

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()  # Set model to evaluation mode

# Function to validate documents before processing
def validate_document(doc):
    """
    Validates a document by checking for null characters and other potential issues.
    """
    if doc.page_content and '\x00' in doc.page_content:
        logging.warning(f"Null character found in document: {doc.metadata.get('source', 'unknown')}")
        return False
    return True

# Function to load documents from file paths (PDF, Text, JSON, Image, Audio)
def load_documents(file_path):
    '''
    Load documents based on the file type (PDF, Text, JSON, Image, Audio), clean the text to remove null characters.
    '''
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif ext == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
    elif ext == '.json':
        loader = JSONLoader(file_path)
        documents = loader.load()
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
        # Store the image file path in metadata
        try:
            documents = [Document(page_content="", metadata={"source": file_path, "type": "image"})]
        except Exception as e:
            st.error(f"Error loading image file {file_path}: {e}")
            logging.error(f"Error loading image file {file_path}: {e}")
            documents = []
    elif ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
        # Transcribe audio file using OpenAI's Whisper
        try:
            model = whisper.load_model("base")
            transcription = model.transcribe(file_path)
            text = transcription['text']
            documents = [Document(page_content=text, metadata={"source": file_path})]
        except Exception as e:
            st.error(f"Error transcribing audio file {file_path}: {e}")
            logging.error(f"Error transcribing audio file {file_path}: {e}")
            documents = []
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Clean the documents to remove null characters and ensure valid UTF-8 encoding
    for doc in documents:
        if doc.page_content:
            # Remove null characters
            doc.page_content = doc.page_content.replace('\x00', '')
            # Normalize encoding
            doc.page_content = codecs.decode(codecs.encode(doc.page_content, 'utf-8', 'ignore'), 'utf-8', 'ignore')
    return documents

# Function to split loaded documents into smaller chunks for processing
def split_documents(documents):
    '''
    Split text documents into smaller chunks for better processing with the LLM.
    '''
    text_documents = [doc for doc in documents if doc.page_content.strip() != ""]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(text_documents)
    # Add non-text documents without splitting
    other_docs = [doc for doc in documents if doc not in text_documents]
    split_docs.extend(other_docs)
    return split_docs

# Function to check and create the database if it doesn't exist
def create_database_if_not_exists(connection_params):
    '''
    Connects to the default 'postgres' database to check and create the target database if it doesn't exist.
    Displays status messages in the Web UI.
    '''
    dbname = connection_params['database']
    user = connection_params['user']
    password = connection_params['password']
    host = connection_params['host']
    port = connection_params['port']

    # Connect to the default 'postgres' database
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=user,
            password=password,
            host=host,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if the database exists
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (dbname,))
        exists = cur.fetchone()
        if not exists:
            st.info(f"Database '{dbname}' does not exist. Creating database...")
            cur.execute(f"CREATE DATABASE {dbname}")
            st.success(f"Database '{dbname}' created successfully.")
        else:
            st.info(f"Database '{dbname}' already exists.")
            logging.info(f"Database '{dbname}' already exists.")

        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        logging.error(f"Error connecting to the database: {e}")

# Function to check and create the pgvector extension
def create_extension_if_not_exists(connection_string):
    '''
    Connects to the target database and creates the pgvector extension if it doesn't exist.
    Displays status messages in the Web UI.
    '''
    try:
        conn = psycopg2.connect(connection_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if the extension exists
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        exists = cur.fetchone()
        if not exists:
            st.info("pgvector extension is not installed. Installing extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            st.success("pgvector extension installed successfully.")
        else:
            st.info("pgvector extension is already installed.")
            logging.info("pgvector extension is already installed.")

        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error checking or creating pgvector extension: {e}")
        logging.error(f"Error checking or creating pgvector extension: {e}")

# Function to vectorize documents using PGVector and appropriate embeddings
def create_vector_store(documents, collection_name, connection_string):
    '''
    Vectorize documents for similarity search using PGVector and appropriate embeddings.
    Displays status messages in the Web UI.
    '''
    try:
        with st.spinner("Creating vector store and embedding documents..."):
            # Separate text and image documents
            text_documents = [doc for doc in documents if doc.page_content.strip() != ""]
            image_documents = [doc for doc in documents if doc.metadata.get('type') == 'image']

            # Initialize vector store
            vector_store = None

            # Process text documents
            if text_documents:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = PGVector.from_documents(
                    documents=text_documents,
                    embedding=embeddings,
                    collection_name=collection_name + "_text",
                    connection_string=connection_string,
                )

            # Process image documents
            if image_documents:
                image_embeddings_list = []
                for doc in image_documents:
                    image_path = doc.metadata.get('source')
                    try:
                        image = Image.open(image_path).convert('RGB')  # Load image from file path
                    except Exception as e:
                        st.error(f"Error loading image {image_path}: {e}")
                        logging.error(f"Error loading image {image_path}: {e}")
                        continue  # Skip this document if image cannot be loaded
                    inputs = clip_processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        embeddings = clip_model.get_image_features(**inputs)
                    embeddings = embeddings.cpu().numpy().flatten()
                    doc.page_content = ""  # Ensure page_content is a string
                    doc.embedding = embeddings
                    image_embeddings_list.append(embeddings)
                if vector_store:
                    vector_store.add_embeddings(
                        documents=image_documents,
                        embeddings=image_embeddings_list,
                    )
                else:
                    vector_store = PGVector.from_embeddings(
                        documents=image_documents,
                        embeddings=image_embeddings_list,
                        collection_name=collection_name + "_image",
                        connection_string=connection_string,
                    )

            # Handle if no documents were processed
            if not vector_store:
                st.warning("No documents were processed for vector store creation.")
                return None

        st.success("Vector store created successfully.")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        logging.error(f"Error creating vector store: {e}")
        raise

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
        # Clean the text
        text = text.replace('\x00', '')
        text = codecs.decode(codecs.encode(text, 'utf-8', 'ignore'), 'utf-8', 'ignore')
        document = Document(page_content=text, metadata={"source": url})
        return [document]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the web page: {e}")
        logging.error(f"Error fetching the web page: {e}")
        return []

# Function to parse the connection string into its components
def parse_connection_string(connection_string):
    '''
    Parses the connection string into its components.
    '''
    import re
    pattern = re.compile(
        r"postgresql://(?P<user>[^:]+):(?P<password>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<database>.+)"
    )
    match = pattern.match(connection_string)
    if match:
        return match.groupdict()
    else:
        raise ValueError("Invalid connection string format.")

# Main Streamlit Application
def main():
    st.set_page_config(page_title="Virtual TA for LLM Agents Course", layout="wide")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Home", "Upload Files", "Ask Questions", "About"])

    # Database Configuration (can be hidden under an expander)
    with st.sidebar.expander("Database Configuration"):
        collection_name = st.text_input("Collection Name", value="my_collection")
        db_user = st.text_input("DB Username", value="postgres")
        db_password = st.text_input("DB Password", value="1234", type="password")
        db_host = st.text_input("DB Hostname", value="localhost")
        db_port = st.text_input("DB Port", value="5432")
        db_name = st.text_input("DB Name", value="mydb")

    # Temperature Setting in Sidebar
    with st.sidebar.expander("Model Settings"):
        temperature = st.slider("Select Temperature:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        st.write(f"Current Temperature: {temperature}")

    # Create connection string
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Parse connection string
    connection_params = parse_connection_string(connection_string)

    # Create the database if it doesn't exist
    create_database_if_not_exists(connection_params)

    # Ensure pgvector extension is installed
    create_extension_if_not_exists(connection_string)

    # Load initial documents from local directories
    def load_initial_documents():
        all_documents = []

        # Directories containing PDF, text, image, and audio files
        pdf_directory = './pdf'
        text_directory = './text'
        image_directory = './images'
        audio_directory = './audio'

        # Load and process all PDF files in the ./pdf directory
        pdf_files = glob.glob(os.path.join(pdf_directory, '*.pdf'))
        for pdf_file in pdf_files:
            logging.info(f"Processing PDF file: {pdf_file}")
            documents = load_documents(pdf_file)
            all_documents.extend(documents)

        # Load and process all text files in the ./text directory
        text_files = glob.glob(os.path.join(text_directory, '*.txt'))
        for text_file in text_files:
            logging.info(f"Processing text file: {text_file}")
            documents = load_documents(text_file)
            all_documents.extend(documents)

        # Load and process all image files in the ./images directory
        image_files = glob.glob(os.path.join(image_directory, '*.*'))
        for image_file in image_files:
            if os.path.splitext(image_file)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                logging.info(f"Processing image file: {image_file}")
                documents = load_documents(image_file)
                all_documents.extend(documents)

        # Load and process all audio files in the ./audio directory
        audio_files = glob.glob(os.path.join(audio_directory, '*.*'))
        for audio_file in audio_files:
            if os.path.splitext(audio_file)[1].lower() in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
                logging.info(f"Processing audio file: {audio_file}")
                documents = load_documents(audio_file)
                all_documents.extend(documents)

        return all_documents

    # Initialize vector store and LLM
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None

    if 'llm' not in st.session_state:
        st.session_state['llm'] = None

    if app_mode == "Home":
        st.title("Virtual TA for Berkeley Large Language Model Agents Course")
        st.write("""
        Welcome to the Virtual Teaching Assistant for the **[CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)** course at UC Berkeley.
        This application allows you to query course materials and receive factual, contextually relevant answers.
        """)
        st.write("Navigate to **Upload Files** to add more documents or to **Ask Questions** based on the knowledge base.")

        # Load initial documents
        all_documents = load_initial_documents()
        if all_documents:
            # Validate documents
            valid_docs = [doc for doc in all_documents if validate_document(doc)]
            invalid_docs = [doc for doc in all_documents if not validate_document(doc)]
            if invalid_docs:
                st.warning(f"Skipped {len(invalid_docs)} documents due to null characters or encoding issues. See log for details.")
            # Split documents
            split_docs = split_documents(valid_docs)
            # Create vector store
            try:
                st.info("Creating vector store with initial documents...")
                vector_store = create_vector_store(split_docs, collection_name, connection_string)
                st.session_state['vector_store'] = vector_store
                st.success("Initial documents loaded and stored in the vector database.")
            except Exception as e:
                st.error(f"Failed to create vector store: {e}")
                logging.error(f"Failed to create vector store: {e}")
        else:
            st.warning("No local documents found in the directories.")

        # Initialize LLAVA model
        st.write("Initializing LLAVA model...")
        try:
            # Adjust the model name based on your hardware capabilities
            with st.spinner("Loading LLAVA model..."):
                llm = Ollama(model="llava-llama3:latest", temperature=temperature)
            st.session_state['llm'] = llm
            st.success("LLAVA model initialized.")
        except Exception as e:
            st.error(f"Failed to initialize LLAVA model: {e}")
            logging.error(f"Failed to initialize LLAVA model: {e}")

    elif app_mode == "Upload Files":
        st.title("Upload Files to Knowledge Base")

        all_documents = []

        # File uploads
        st.subheader("Upload Files (PDF, Text, Images, Audio)")
        uploaded_files = st.file_uploader("Upload files", type=['pdf', 'txt', 'png', 'jpg', 'jpeg', 'bmp', 'gif', 'wav', 'mp3', 'm4a', 'flac', 'ogg'], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                file_path = os.path.join("temp_uploads", uploaded_file.name)
                os.makedirs("temp_uploads", exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                # Load and process the file
                documents = load_documents(file_path)
                all_documents.extend(documents)
                st.success(f"Processed file: {uploaded_file.name}")

        # Input URLs
        st.subheader("Input URLs to Scrape")
        url_input = st.text_area("Enter URLs (one per line)")
        if st.button("Scrape and Add URLs"):
            urls = [line.strip() for line in url_input.split('\n') if line.strip()]
            for url in urls:
                st.info(f"Scraping URL: {url}")
                scraped_documents = scrape_web_page(url)
                if scraped_documents:
                    all_documents.extend(scraped_documents)
                    st.success(f"Scraped and processed URL: {url}")
                else:
                    st.error(f"Failed to scrape URL: {url}")

        # Process and store documents
        if all_documents:
            st.header("Processing Documents")
            # Validate documents
            valid_docs = [doc for doc in all_documents if validate_document(doc)]
            invalid_docs = [doc for doc in all_documents if not validate_document(doc)]
            if invalid_docs:
                st.warning(f"Skipped {len(invalid_docs)} documents due to null characters or encoding issues. See log for details.")
            if valid_docs:
                with st.spinner("Splitting and embedding documents..."):
                    split_docs = split_documents(valid_docs)
                    if st.session_state['vector_store'] is None:
                        try:
                            st.info("Creating new vector store...")
                            st.session_state['vector_store'] = create_vector_store(split_docs, collection_name, connection_string)
                            st.success("Documents have been processed and added to the vector database.")
                        except Exception as e:
                            st.error(f"Failed to create vector store: {e}")
                            logging.error(f"Failed to create vector store: {e}")
                    else:
                        # Add documents to existing vector store
                        try:
                            st.info("Adding documents to existing vector store...")
                            st.session_state['vector_store'].add_documents(split_docs)
                            st.success("Documents have been processed and added to the vector database.")
                        except Exception as e:
                            st.error(f"Failed to add documents to vector store: {e}")
                            logging.error(f"Failed to add documents to vector store: {e}")
            else:
                st.warning("No valid documents to process after validation.")
        else:
            st.warning("No new documents to process.")

    elif app_mode == "Ask Questions":
        st.title("Ask Questions to the Virtual TA")

        llm = st.session_state.get('llm', None)
        vector_store = st.session_state.get('vector_store', None)

        if llm and vector_store:
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

            # Update LLM temperature
            llm.temperature = temperature

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
            )

            question = st.text_input("Enter your question:")
            image_input = st.file_uploader("Upload an image to include in your question (optional)", type=['png', 'jpg', 'jpeg', 'bmp', 'gif'])

            if st.button("Get Answer"):
                if question or image_input:
                    with st.spinner("Generating answer..."):
                        try:
                            if image_input:
                                # Read image
                                image = Image.open(image_input).convert('RGB')
                                # Process image using CLIP
                                inputs = clip_processor(images=image, return_tensors="pt")
                                with torch.no_grad():
                                    image_embedding = clip_model.get_image_features(**inputs)
                                image_embedding = image_embedding.cpu().numpy().flatten()
                                # Use the image embedding in your retrieval or prompt
                                # Customize this part based on how LLAVA accepts image inputs
                            #     answer = qa_chain.run({"question": question, "image_embedding": image_embedding})
                            # else:
                            #     answer = qa_chain.run(question)
                                answer = qa_chain.run({"query": question, "image_embedding": image_embedding})
                            else:
                                answer = qa_chain.run({"query": question})
                            st.subheader("Answer:")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Error generating answer: {e}")
                            logging.error(f"Error generating answer: {e}")
                else:
                    st.warning("Please enter a question or upload an image.")
        else:
            st.error("The system is not fully initialized. Please ensure that the vector store and LLAVA model are initialized.")

    elif app_mode == "About":
        st.title("About the Virtual TA Application")
        st.write("""
        This application serves as a Virtual Teaching Assistant for the **CS294/194-196 Large Language Model Agents** course at UC Berkeley.
        It leverages advanced language models and vector databases to provide accurate and contextually relevant answers based on uploaded documents and course materials.
        """)

    else:
        st.error("Invalid app mode selected.")

if __name__ == "__main__":
    main()
