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

Students can interact with the virtual TA through Edstem, and it will guide them by offering insights into course topics, helping with understanding complex concepts, and ensuring they remain engaged with the course material.

This AI-powered TA ensures that students can access timely and accurate information, enhancing their learning experience in the rapidly evolving field of large language model agents.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.


For any inquiries, feel free to use the virtual TA and explore the topics discussed in the course!
