# -*- coding: utf-8 -*-
"""KnowledgeBase_RAG_with_PGVector_and_WebUI.py"""

import os
import glob
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Updated imports from langchain_community
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document

import warnings

# Import Streamlit for the Web UI
import streamlit as st

# Import codecs and logging for text processing and logging
import codecs
import logging

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, filename='document_processing.log')

# Function to load documents from file paths (PDF, Text, JSON)
def load_documents(file_path):
    '''
    Load documents based on the file type (PDF, Text, JSON), clean the text to remove null characters.
    '''
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext.lower() == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif ext.lower() == '.json':
        loader = JSONLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    documents = loader.load()

    # Clean the documents to remove null characters and ensure valid UTF-8 encoding
    for doc in documents:
        if doc.page_content:
            # Remove null characters
            doc.page_content = doc.page_content.replace('\x00', '')
            # Normalize encoding
            doc.page_content = codecs.decode(codecs.encode(doc.page_content, 'utf-8', 'ignore'), 'utf-8', 'ignore')
    return documents

# Function to validate documents before processing
def validate_document(doc):
    if '\x00' in doc.page_content:
        logging.warning(f"Null character found in document: {doc.metadata.get('source', 'unknown')}")
        return False
    return True

# Function to split loaded documents into smaller chunks for processing
def split_documents(documents):
    '''
    Split documents into smaller chunks for better processing with the LLM.
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
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
            # Create the database
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
            # Create the extension
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

# Function to vectorize documents using PGVector and HuggingFace embeddings
def create_vector_store(documents, collection_name, connection_string):
    '''
    Vectorize documents for similarity search using PGVector and HuggingFace embeddings.
    Displays status messages in the Web UI.
    '''
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        with st.spinner("Creating vector store and embedding documents..."):
            vector_store = PGVector.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                connection_string=connection_string,
            )
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

        # Directories containing PDF and text files
        pdf_directory = './pdf'
        text_directory = './text'

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
        st.write("Loading initial documents from local directories...")
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
            st.warning("No local documents found in './pdf' or './text' directories.")

        # Initialize LLaMA model
        st.write("Initializing LLaMA model...")
        try:
            # Adjust the model name based on your hardware capabilities
            with st.spinner("Loading LLaMA model..."):
                llm = Ollama(model="llava-llama3:latest", temperature=temperature)
            st.session_state['llm'] = llm
            st.success("LLaMA model initialized.")
        except Exception as e:
            st.error(f"Failed to initialize LLaMA model: {e}")
            logging.error(f"Failed to initialize LLaMA model: {e}")

    elif app_mode == "Upload Files":
        st.title("Upload Files to Knowledge Base")

        all_documents = []

        # File uploads
        st.subheader("Upload PDF and Text Files")
        uploaded_files = st.file_uploader("Upload PDF or Text files", type=['pdf', 'txt'], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
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
            if st.button("Get Answer"):
                if question:
                    with st.spinner("Generating answer..."):
                        try:
                            answer = qa_chain.run(question)
                            st.subheader("Answer:")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Error generating answer: {e}")
                            logging.error(f"Error generating answer: {e}")
                else:
                    st.warning("Please enter a question.")
        else:
            st.error("The system is not fully initialized. Please ensure that the vector store and LLaMA model are initialized.")

    elif app_mode == "About":
        st.title("About the Virtual TA Application")

        # Display the provided information along with the explanation of the "temperature" parameter
        st.markdown("""
        # Virtual TA for Berkeley Large Language Model Agents Course

        ## Overview

        This project serves as a **virtual Teaching Assistant (TA)** for the **[CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)** course at UC Berkeley. It leverages a Retrieval-Augmented Generation (RAG) system, built using LangChain and PGVector, to create a knowledge base that assists students by providing factual, contextually relevant answers to course-related questions.

        The system allows users to load and query course materials, including scientific papers, lecture notes, and other documents. It is powered by HuggingFace embeddings and uses PostgreSQL with PGVector for efficient semantic search. The local LLaMA model (Ollama) is integrated to generate responses based on the most relevant documents retrieved from the knowledge base. This RAG system acts as a virtual TA, available to help students understand and engage with course content.

        ## Features
        - Acts as a virtual TA for the Berkeley CS294/194-196 course.
        - Load documents from PDF, text, or JSON files, including lecture materials and scientific papers.
        - Split documents into manageable chunks for efficient processing and search.
        - Vectorize and store documents using HuggingFace embeddings in a PostgreSQL database with PGVector.
        - Query the knowledge base using natural language questions, with responses generated using the local LLaMA model.
        - Scrape web pages and integrate their content into the knowledge base, dynamically expanding the system's resources.
        - Adjust the model's creativity using the **temperature** setting.

        ## Model Temperature Setting

        The **temperature** setting controls how creative or deterministic the model's responses are. It adjusts the level of randomness in the language model's output.

        - **Low temperature (close to 0)**: Makes the model more deterministic, resulting in predictable and focused outputs. This is useful for tasks where precision and consistency are crucial.
        - **High temperature (close to 1 or higher)**: Introduces more randomness, leading to diverse and creative responses. This can be beneficial for brainstorming or generating novel ideas.

        You can adjust the temperature setting in the sidebar under **Model Settings** to suit your preference for the model's responses.

        ## Hardware Recommendations

        Based on your hardware specifications:

        - **Memory**: 120GB of RAM
        - **GPUs**: Two GPUs

        You have substantial resources to run larger language models. It's recommended to use models that can leverage multi-GPU setups and high memory availability.

        ### Suggested Models:

        - **LLaMA 2 (70B parameters)**: A powerful model that provides high-quality responses. With your hardware, you should be able to run this model efficiently, especially if your GPUs have sufficient VRAM.
        - **GPT-NeoX-20B**: An open-source model that balances performance and resource requirements.
        - **Falcon-40B**: Another large model known for its performance in various tasks.

        Ensure that your GPUs have enough VRAM to load the model weights. If VRAM is a limitation, consider using model parallelism or techniques like gradient checkpointing to manage memory usage.

        **Note**: Running large models requires appropriate software configurations, such as optimized deep learning frameworks (e.g., PyTorch with CUDA support) and possibly model parallelism libraries.

        ## Course Description

        The **[Berkeley CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)** course explores the development and application of large language models (LLMs) in various domains, such as code generation, robotics, web automation, and scientific discovery. The course also covers the foundational aspects of LLMs, their abilities, and the infrastructure required for agent development, as well as the ethical considerations, privacy issues, and future directions of LLMs.

        ### Topics Covered:
        - **LLM Fundamentals**: Task automation and agent infrastructure.
        - **LLM Applications**: Code generation, web automation, multimodal agents, and robotics.
        - **Limitations and Future Directions**: Privacy, safety, ethics, and multi-agent collaboration.

        ### Syllabus

        | Lecture # | Date     | Topic                                     | Guest Lecture                             |
        |-----------|----------|-------------------------------------------|-------------------------------------------|
        | 1         | Sept 9   | LLM Reasoning                             | Denny Zhou, Google DeepMind               |
        | 2         | Sept 16  | LLM agents: brief history and overview    | Shunyu Yao, OpenAI                        |
        | 3         | Sept 23  | Agentic AI Frameworks & AutoGen           | Chi Wang, AutoGen-AI                      |
        | 4         | Sept 30  | Enterprise trends for generative AI       | Burak Gokturk, Google                     |
        | 5         | Oct 7    | Compound AI Systems & the DSPy Framework  | Omar Khattab, Databricks                  |
        | 6         | Oct 14   | Agents for Software Development           | Graham Neubig, Carnegie Mellon University |
        | 7         | Oct 21   | AI Agents for Enterprise Workflows        | Nicolas Chapados, ServiceNow              |
        | 8         | Oct 28   | Neural Networks with Symbolic Decision-Making | Yuandong Tian, Meta AI (FAIR)         |
        | 9         | Nov 4    | Foundation Agent                          | Jim Fan, NVIDIA                           |
        | 10        | Nov 18   | Cybersecurity, agents, and open-source    | Percy Liang, Stanford University          |
        | 11        | Dec 2    | LLM Agent Safety                          | Dawn Song, UC Berkeley                    |

        ## Enrollment and Grading

        This is a variable-unit course:
        - **1 unit**: Submit a lecture summary article.
        - **2 units**: Submit a lab assignment and a written project report (surveying LLM topics).
        - **3 units**: Include a coding component in the project.
        - **4 units**: Include a significant implementation with potential real-world impacts or intellectual contributions.

        ## Virtual TA for LLM Agents Course

        This **KnowledgeBase RAG with PGVector** is designed to serve as the virtual TA for the **[CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)** course. Given the course's focus on scientific papers and web-based content, the virtual TA assists students by providing factual, detailed answers based on course materials. It is specifically built to respond to student queries related to lectures, readings, and assignments using the RAG pipeline.

        Students can interact with the virtual TA and it will guide them by offering insights into course topics, helping with understanding complex concepts, and ensuring they remain engaged with the course material.

        This AI-powered TA ensures that students can access timely and accurate information, enhancing their learning experience in the rapidly evolving field of large language model agents.

        ## License

        This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

        For any inquiries, feel free to use the virtual TA and explore the topics discussed in the course!
        """)

    else:
        st.error("Invalid app mode selected.")

if __name__ == "__main__":
    main()
