# Virtual TA for Berkeley Large Language Model Agents Course

## Overview

This project provides a **virtual Teaching Assistant (TA)** for the **[CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)** course at UC Berkeley. It utilizes a **multi-modal Retrieval-Augmented Generation (RAG) system** that processes **image, text, and speech** inputs, built using LangChain and PGVector, to create a knowledge base that helps students by providing factual, contextually relevant answers to course-related questions.

The system allows users to load and query course materials, such as scientific papers, lecture notes, and other documents. It is powered by HuggingFace embeddings and uses PostgreSQL with PGVector for efficient semantic search. The local LLaMA model (Ollama) generates responses based on the most relevant documents retrieved from the knowledge base. This RAG system acts as a virtual TA to assist students in engaging with course content.

## Virtual TA for LLM Agents Course

The **KnowledgeBase RAG with PGVector** serves as the virtual TA for the **[CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)** course. The system assists students by providing factual, detailed answers based on course materials, which include lectures, readings, and assignments. Using the RAG pipeline, students can query the virtual TA for insights into course topics and receive contextually accurate information.

This AI-powered TA enhances students' learning experience, ensuring they stay engaged with the rapidly evolving field of large language model agents.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/virtual-ta-llm-agents.git
   cd virtual-ta-llm-agents
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup PostgreSQL with PGVector:
   - Ensure PostgreSQL is installed and running.
   - Follow the [PGVector installation guide](https://github.com/pgvector/pgvector) to enable vector support in PostgreSQL.

4. Set up the virtual environment:
   ```bash
   source setup.sh
   ```

## Running the Application

1. After setting up the environment, run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

For any inquiries, feel free to use the virtual TA and explore the topics discussed in the course!

Contact: mmemari@uvu.edu
