# Retrieval-Augmented Generation (RAG) for Local Data Processing
### Project Overview

This repository is dedicated to building a **custom Retrieval-Augmented Generation (RAG)** system that operates entirely on your **local** machine. By leveraging the **Ollama** platform, we aim to create an **AI solution** similar to **ChatGPT** that can process and analyze **text** data such as patient records for **Customer Relationship Management (CRM)** and customer review **sentiment analysis**—all without requiring an **internet** connection.

Our primary objective is to develop a local AI system that effectively combines retrieval (searching for relevant information within your datasets) and generation (producing coherent and context-aware responses) to assist in tasks like data analysis, report generation, and decision support. By processing all data **locally**, we ensure strict **confidentiality** and compliance with **privacy** regulations, making it ideal for handling **sensitive** information like **patient** records and **proprietary** customer feedback.

This project not only provides a practical solution for **managing** and **interpreting** your own data but also serves as an **educational** resource to understand how **RAG** systems work. Users can learn how to set up, customize, and operate their own local RAG systems to suit specific data processing needs while maintaining complete **control** over their data.

## Key Concepts

### 1. Retrieval-Augmented Generation (RAG)
**RAG** enhances large language models by retrieving relevant information from external data sources, such as documents, databases, or a vector database. The system represents local documents as embeddings (vectors), which allows the model to find information based on the input query.

RAG is particularly effective when:
- The language model lacks sufficient context or knowledge about a particular subject.
- The data being queried is large, but only certain relevant sections are needed.
- The queries should be answered using specific or recent data from local files.

### 2. Vector Database
A **vector database** is a specialized database designed to store and retrieve high-dimensional vectors, also known as embeddings. In this RAG setup, documents are converted into embeddings (vector representations) and stored in the vector database. When a query is made, it is also converted into a vector, and the vector database quickly retrieves the closest matching document embeddings based on semantic similarity.

Vector databases are essential for scaling retrieval tasks, as they allow efficient search across large datasets by finding the most relevant vectors in a multi-dimensional space. Here, Ollama leverages **embeddings** to represent document meaning and stores them in an internal vector database for efficient retrieval.

### 3. Embeddings
Embeddings are vector representations of text that capture the semantic meaning of words or documents. The system uses Ollama's `mxbai-embed-large` model to create high-quality embeddings for the local documents. These embeddings are stored in a vector database, enabling fast and accurate retrieval based on user queries.

Embeddings serve as the backbone of the RAG system. By transforming text into a numerical format, the system can quickly compare and retrieve the most relevant documents or portions of data.

### 4. Local LLMs with Ollama
**Ollama** is an open-source platform that allows the running of large language models (LLMs) directly on your machine, without needing cloud-based services. This local execution ensures data privacy and minimizes latency while still maintaining the powerful capabilities of models like **Llama3**.

## Setup

The following steps will help you set up the Local RAG system on your laptop. This setup is designed to be lightweight and straightforward, making it ideal for educational purposes or prototyping.

### Step 1: Clone the Repository
First, clone this repository to your local machine:

```bash
git clone https://github.com/memari-majid/easy-local-rag.git
```

Go to the repository downloaded in your local machine:

```bash
cd easy-local-rag
```
### Step 2: Install Python Dependencies
Install the required Python packages by running the following command:

```bash
pip install -r requirements.txt
```
These dependencies include tools for managing language models, handling documents, and generating embeddings.

### Step 3: Install Ollama
Ollama is the core platform for running the local language models. Download and install it from the official website:
[Download Ollama](https://ollama.com/)
Once installed, Ollama will allow access to a variety of optimized language models for local use.

### Step 4: Pull the Required Models
You will need to download the specific models that power the RAG system. Use the following commands to pull the necessary models:

Llama3 for general-purpose language generation:

```bash
ollama pull llama3
```

mxbai-embed-large for generating high-quality embeddings from documents:

```bash
ollama pull mxbai-embed-large
```

These models will be stored locally, allowing the entire system to run offline.

### Step 5: Upload Your Documents
The system supports various document formats, including .pdf, .txt, and .json. To upload your local documents, run the following script:

```bash
python upload.py
```
This script processes your documents and converts them into embeddings, storing them locally for future queries. The embeddings are added to the vector database, enabling efficient retrieval when queried.

### Step 6: Query Your Documents
Once your documents are uploaded and the embeddings are generated, you can start querying your local data using the RAG system.

To run the system with query rewriting (useful for vague or unclear queries), use:

```bash
python localrag.py
```

For direct control over the results without query rewriting, use:

```bash
python localrag_no_rewrite.py
```

Both options allow you to ask natural language questions, and the system will retrieve and generate responses based on the relevant sections of your local documents.