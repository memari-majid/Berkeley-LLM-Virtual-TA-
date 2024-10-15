# KnowledgeBase RAG with PGVector

## Overview
This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using LangChain and PGVector to create a semantic knowledge base. The system allows users to load documents, vectorize them using HuggingFace embeddings, and store them in PostgreSQL with PGVector for efficient semantic search. The knowledge base can then be queried to provide responses based on the most relevant documents using a local LLaMA model (Ollama).

The system also includes functionality for scraping web pages and integrating their content into the knowledge base, allowing for dynamic and flexible document inclusion. By connecting a Retriever, Prompt, and LLM, the project demonstrates the creation of a complete RAG pipeline.

## Features
- Load documents from PDF, text, or JSON files.
- Split documents into smaller chunks for better processing.
- Vectorize and store documents in a PostgreSQL database using PGVector.
- Create a RAG pipeline using a local LLaMA model (Ollama).
- Scrape content from web pages and add it to the knowledge base.
- Query the knowledge base with natural language questions.

