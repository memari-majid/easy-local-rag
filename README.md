# Retrieval-Augmented Generation (RAG) for Local Data Processing
### Project Overview

This repository focuses on creating a **local Retrieval-Augmented Generation (RAG)** system designed to understand your queries through a **Large Language Model (LLM)**, search your local database, and semantically retrieve relevant information. By integrating the **Ollama** platform, this project enables an advanced AI system similar to **ChatGPT**, but it operates entirely on your **local machine** without requiring an **internet** connection.

The local RAG system leverages the LLM's ability to **comprehend** the nuances of your **query**, enabling it to search through your dataset and extract **semantically** related information. Unlike a traditional database query, which requires **precise** **keywords** or structured queries (such as SQL), the RAG system allows for natural language input. This means that instead of returning **exact keyword matches**, the RAG system can **interpret** the **context** and **intent** behind your query, pulling information that is **semantically relevant** even if it **doesn NOT match exact keywords**.

For example, in a **regular database query**, asking for "customer complaints about delivery delays" might only return entries explicitly tagged with "delivery delay." In contrast, a **RAG-based system** can understand **synonyms**, **variations**, or **implicit** mentions of the issue, such as "shipment took too long" or "late delivery." This makes it significantly more powerful when searching through unstructured or loosely structured data like **customer reviews**, **patient records**, or **emails**.

By performing all tasks **locally**, the system guarantees **data privacy** and ensures compliance with **confidentiality** regulations, making it ideal for handling **sensitive information** such as **patient records** or **proprietary customer feedback**. Additionally, the combination of retrieval and generation offers robust **data analysis**, **report generation**, and **decision support** capabilities.

Integrating a **RAG system** into your database infrastructure can dramatically enhance the ability to extract actionable insights from large datasets. Instead of relying on rigid, predefined query patterns, a RAG system empowers users to engage with their data in a more **intuitive** and **context-aware** manner. This enables more **flexible exploration** of data, more accurate **information retrieval**, and the ability to generate **coherent explanations** for complex queries—all while maintaining full **control** over data workflows and privacy.

This project not only provides a powerful tool for managing and interpreting your own data but also serves as a learning resource for building customized RAG systems. Users can explore how the system semantically searches their databases and generates context-aware responses, enhancing the capability of traditional database management.

## Key Concepts

### 1. Retrieval-Augmented Generation (RAG)
**RAG** enhances large language models by retrieving relevant information from external data sources, such as documents, databases, or a vector database. The system represents local documents as embeddings (vectors), which allows the model to find information based on the input query.

RAG is particularly effective when:
- The language model lacks sufficient context or knowledge about a particular subject.
- The data being queried is large, but only certain relevant sections are needed.
- The queries should be answered using specific or recent data from local files.

### 2. Vector Database (Using FAISS)
A **vector database** is used to store and retrieve high-dimensional vectors, or **embeddings**, that represent the **semantic** meaning of documents. In this project, we use **FAISS (Facebook AI Similarity Search)**, which allows us to store document embeddings and retrieve the most semantically similar documents based on the **query** embedding.

After you upload documents (PDF, text, or JSON), the system converts these documents into embeddings using the **Ollama mxbai-embed-large** model. These **embeddings** are then indexed using **FAISS**, allowing for efficient **semantic search** based on the meaning of the user's query.

When a user asks a question, the system converts the query into an embedding and searches for the most similar document embeddings using FAISS. This allows the system to retrieve relevant information, even if the words in the query do not match the exact words in the documents, as the search is based on meaning rather than keywords.

#### Persistent Storage of Embeddings and FAISS Index
To improve efficiency, the embeddings generated from the uploaded documents and the FAISS index are **saved to disk**. This means that if you re-run the program later, the embeddings and index will be **loaded** instead of re-generated, saving time.

- **FAISS Index**: Saved as `faiss_index.bin`.
- **Embeddings**: Saved as `vault_embeddings.npy`.

If new documents are uploaded, the system will regenerate the embeddings and re-index the entire set of documents.

### 3. Embeddings
Embeddings are vector representations of text that capture the semantic meaning of words or documents. The system uses Ollama's `mxbai-embed-large` model to create high-quality embeddings for the local documents. These embeddings are stored in a vector database, enabling fast and accurate retrieval based on user queries.

Embeddings serve as the backbone of the RAG system. By transforming text into a numerical format, the system can quickly compare and retrieve the most relevant documents or portions of data.

### 4. Local LLMs with Ollama
**Ollama** is an open-source platform that allows the running of large language models (LLMs) directly on your machine, without needing cloud-based services. This local execution ensures data privacy and minimizes latency while still maintaining the powerful capabilities of models like **llama3**.

## Setup

The following steps will help you set up the Local RAG system on your laptop. This setup is designed to be lightweight and straightforward, making it ideal for educational purposes or prototyping.

### Step 1: Clone the Repository
First, clone this repository to your local machine:

```bash
git clone https://github.com/memari-majid/local_rag.git
```

Go to the repository downloaded in your local machine:

```bash
cd local_rag
```

### Step 2: Install Package Manager
To manage Python environments more effectively, we recommend using **Miniconda**. Follow these steps to install Miniconda:

1. **Download Miniconda**: Go to the [Miniconda Installation page](https://docs.conda.io/en/latest/miniconda.html) and download the installer suitable for your operating system (Windows, macOS, Linux).

2. **Run the Installer**:
   - On **Windows**, open the installer and follow the installation instructions.
   - On **macOS** and **Linux**, run the following commands in your terminal (replace the installer name with the appropriate one for your OS):

   For Linux
   ```bash
   
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

   For macOS
   ```bash
   bash Miniconda3-latest-MacOSX-x86_64.sh
   ```
   Follow the Prompts: During the installation process, follow the prompts to install Miniconda and allow the installer to initialize the conda environment.

3. **Create a New Conda Environment with Python 3.9**
Once Miniconda is installed, create a new environment with Python 3.9 using the following commands:

Create a new conda environment with Python 3.9
```bash
conda create --name rag python=3.9
```
Activate the environment
```bash
conda activate rag
```

### Step 3: Install Python Dependencies
Install the required Python packages by running the following command:

```bash
pip install -r requirements.txt
```
These dependencies include tools for managing language models, handling documents, and generating embeddings.

### Step 4: Install Ollama


Ollama is the core platform for running the local language models.

#### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

#### Windows

[Download](https://ollama.com/download/OllamaSetup.exe)

#### Linux Install

To install Ollama, run the following command:

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

##### Linux Manual install

Download and extract the package:

```shell
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz
```

Start Ollama:

```shell
ollama serve
```

#### Docker

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` is available on Docker Hub.


Once installed, Ollama will allow access to a variety of optimized language models for local use.

### Step 5: Pull the Required Models
You will need to download the specific models that power the RAG system. 
#### Models and Memory Requirements

The following table outlines various **Large Language Models (LLMs)** with their corresponding number of parameters and memory requirements. **llama** models, especially the Llama 3 series developed by Meta, are highly advanced models **optimized** for **Natural Language Processing (NLP)** tasks. Llama 3 provides improvements in performance, speed, and **efficiency** compared to earlier versions. Different versions of the llama model (with varying numbers of parameters) require varying amounts of memory (RAM) to run effectively.

For example, the **llama 3.1, 8B parameter** model requires **4.7 GB of memory**, which would be a suitable candidate to run **locally** on your **laptop** if you have sufficient RAM. This model offers a balance between **performance** and **memory** footprint, making it practical for local use on systems with moderate hardware capabilities.

| **Model**                | **Parameters** | **Memory Size** | **Ollama Command**             |
|--------------------------|----------------|-----------------|--------------------------------|
| Moondream 2              | 1.4B           | 829 MB          | `ollama run moondream`         |
| Gemma 2                  | 2B             | 1.6 GB          | `ollama run gemma2:2b`         |
| Phi 3 Mini               | 3.8B           | 2.3 GB          | `ollama run phi3`              |
| Code Llama               | 7B             | 3.8 GB          | `ollama run codellama`         |
| Llama 2 Uncensored       | 7B             | 3.8 GB          | `ollama run llama2-uncensored` |
| Mistral                  | 7B             | 4.1 GB          | `ollama run mistral`           |
| Neural Chat              | 7B             | 4.1 GB          | `ollama run neural-chat`       |
| Starling                 | 7B             | 4.1 GB          | `ollama run starling-lm`       |
| LLaVA                    | 7B             | 4.5 GB          | `ollama run llava`             |
| **Llama 3.1**            | **8B**         | **4.7 GB**      | **`ollama run llama3.1`**      |
| Solar                    | 10.7B          | 6.1 GB          | `ollama run solar`             |
| Gemma 2                  | 9B             | 5.5 GB          | `ollama run gemma2`            |
| Phi 3 Medium             | 14B            | 7.9 GB          | `ollama run phi3:medium`       |
| Gemma 2                  | 27B            | 16 GB           | `ollama run gemma2:27b`        |
| Llama 3.1                | 70B            | 40 GB           | `ollama run llama3.1:70b`      |
| Llama 3.1                | 405B           | 231 GB          | `ollama run llama3.1:405b`     |


When selecting a llama model, it is important to consider the available memory and hardware capabilities, as larger models require more memory but deliver better performance for complex tasks.

Use the following commands to pull the necessary models:

```bash
ollama pull llama3:7b
```

#### Generating High-Quality Embeddings from Documents

The `mxbai-embed-large` model is designed to generate high-quality embeddings from textual documents. Embeddings are vector representations of text that capture the **semantic** meaning of words or sentences. These embeddings are essential for tasks like similarity search, document clustering, and retrieval-augmented generation (RAG) systems.

By leveraging a large embedding model like `mxbai-embed-large`, you can ensure that your documents are encoded into dense vectors that preserve their **semantic** richness and **contextual** information, which improves the performance of downstream tasks such as search, recommendation, or summarization.

To pull the `mxbai-embed-large` model using **ollama**, use the following command:

mxbai-embed-large for generating high-quality embeddings from documents:

```bash
ollama pull mxbai-embed-large
```

These models will be stored locally, allowing the entire system to run offline.


### Step 6: Upload Your Documents

The `upload.py` script allows users to upload documents in **PDF**, **TXT**, and **JSON** formats and append their contents to a file called `vault.txt`. This tool is designed to handle documents by converting them into chunks of text, making them ready for further processing.

#### Key Features of `upload.py`:

- **PDF to Text Conversion**: Converts PDF files to text and normalizes the text by removing unnecessary whitespace. The text is split into chunks (up to 1000 characters) and appended to `vault.txt`, with each chunk on a new line.

- **TXT File Upload**: Reads text from `.txt` files, normalizes whitespace, and splits the content into 1000-character chunks. The text is then appended to `vault.txt`, similar to the PDF functionality.

- **JSON File Upload**: Parses JSON files and flattens the data into a single string. It then processes the content by normalizing whitespace and splitting it into chunks before appending it to `vault.txt`.

#### How the Script Works:

1. **User Interface**: A simple GUI is provided using `tkinter`, where buttons allow users to select and upload their files.
   
2. **PDF Handling**: When a user selects a PDF file, the script extracts the text from each page, normalizes it, and splits it into chunks of up to 1000 characters.
   
3. **Text File Handling**: When a `.txt` file is uploaded, the text is similarly cleaned and split into chunks of up to 1000 characters before being added to `vault.txt`.
   
4. **JSON File Handling**: For JSON files, the script flattens the entire structure into a single text string, normalizes whitespace, and then splits the content into chunks before appending to `vault.txt`.

#### Example Usage:

To upload files using the script, simply run the script and select the file format you wish to upload:

```bash
python upload.py
```
This will open a GUI where you can choose a PDF, TXT, or JSON file. The selected file will be processed, and its content will be appended to `vault.txt` in chunks.

**Adding New Document Types:**
If you want to extend the functionality of upload.py to support additional file types, you can modify the script by adding new functions similar to upload_txtfile or upload_jsonfile.

This script makes it easy to gather documents from various formats into a central text file, ready for processing or analysis.

### Step 7: Query Your Documents

The `localrag.py` script provides a local environment for running Retrieval-Augmented Generation (RAG) using a vault of documents. This script allows you to query a set of local documents and retrieve relevant context for answering questions. It integrates with **Ollama** to generate embeddings and retrieve context based on user input. Additionally, it utilizes **PyTorch** to handle similarity calculations and embeddings.

#### Key Features of `localrag.py`:

- **Document Retrieval**: The script reads from a local text file (`vault.txt`), generates embeddings for the contents, and uses cosine similarity to retrieve relevant text based on the user's query.
  
- **Ollama Integration**: The script uses **Ollama's** embedding and completion models, such as `mxbai-embed-large` for generating embeddings and `llama3` for answering user queries based on the context.
  
- **Query Rewriting**: When a user asks a question, the script can rewrite the query by referencing recent conversation history, improving the query to retrieve better results without changing its original intent.

#### How the Script Works:

1. **Loading Documents**: 
   The script reads a text file named `vault.txt` containing the documents you want to use for context retrieval. Each line of the file is treated as a separate chunk of text, and embeddings are generated for these chunks using the **mxbai-embed-large** model.

2. **Generating Embeddings**: 
   Once the vault is loaded, the script creates embeddings for each chunk using Ollama's embedding model. These embeddings are stored as tensors and used for similarity searches.

3. **User Interaction**:
   - The user is prompted to input a query. If context from the vault is relevant, the script retrieves the top matching chunks based on cosine similarity.
   - If there is sufficient conversation history, the script rewrites the user's query using the most recent context.
   - The query, along with the relevant context, is passed to **Ollama's language model** (e.g., `llama3`) for generating a response.

4. **Conversation Loop**:
   The script enters an interactive loop where the user can continually input queries and receive responses. The conversation history is updated with each interaction to improve context.

#### Example Usage:

To run the `localrag.py` script, use the following command in your terminal:

```bash
python localrag.py
```
This will initiate the conversation loop. The system will use the llama3 model by default for processing your queries. You can specify a different model by passing the --model argument.

**Vault File:**
Ensure that your `vault.txt` is properly formatted, with each document or chunk of text on a new line. The script will generate embeddings for each line and use them to find the most relevant context for your queries.

**Example Commands:**
   - **Ask a Query:** Type a question and the script will retrieve the most relevant content from vault.txt, rewriting your query if needed, and provide a response.

   - **Quit the Script:** To exit the conversation, simply type quit.

**Configuration:**
   - **Ollama API:** The script is configured to interact with an Ollama API running locally on http://localhost:11434/v1. Ensure that the API is running before executing the script.

   - **Embedding Model:** By default, the script uses the mxbai-embed-large model to generate document embeddings.

This script is highly flexible for building local RAG systems, enabling users to query a custom set of documents and receive intelligent, context-aware responses.

### Using `localrag_no_rewrite.py` for Retrieval-Augmented Generation (RAG)

The `localrag_no_rewrite.py` script is a simplified version of a Retrieval-Augmented Generation (RAG) system that allows you to query a set of local documents and retrieve relevant context without rewriting the user's query. This script uses **Ollama** for generating embeddings and answering user queries by pulling relevant context from a local "vault" of documents.

#### Key Features of `localrag_no_rewrite.py`:

- **No Query Rewriting**: Unlike other RAG systems, this script does not rewrite the user's query before retrieving relevant context. Instead, it directly uses the user's input for context retrieval.
  
- **Document Retrieval**: The script reads from a local text file (`vault.txt`), generates embeddings for the contents using **Ollama's** embedding model, and retrieves the most relevant chunks based on the user's query.
  
- **Ollama Integration**: The script utilizes **Ollama's** API for both embedding generation (e.g., `mxbai-embed-large`) and text generation (e.g., `llama3` or other specified models).
  
- **Cosine Similarity Matching**: The system uses cosine similarity to match the user's input with the most relevant chunks of text from the vault, providing highly relevant context for answering queries.

#### How the Script Works:

1. **Loading Documents**: 
   The script reads a file named `vault.txt` containing chunks of documents or text lines. Each line in the file is treated as a document, and embeddings are generated for each chunk using the **mxbai-embed-large** model.

2. **Generating Embeddings**: 
   Embeddings are generated for the vault content using **Ollama's** embedding model. These embeddings are stored as tensors and used to compute cosine similarity for relevance matching.

3. **User Interaction**:
   - The user provides a query. The system retrieves the top relevant chunks of text from the vault based on cosine similarity.
   - The relevant context is concatenated with the user’s input and sent to **Ollama's** language model (e.g., `llama3`) to generate a response.

4. **Conversation Loop**:
   The script continuously prompts the user for input, retrieves context, and provides responses in a loop. The conversation history is maintained to allow continuity in the dialogue.

#### Example Usage:

To run the `localrag_no_rewrite.py` script, use the following command in your terminal:

```bash
python localrag_no_rewrite.py
```
This will initiate the conversation loop. By default, the system uses the llama3 model, but you can specify another model using the --model argument.

**Vault File:**
Ensure that your vault.txt is properly formatted, with each document or chunk of text on a new line. The script will generate embeddings for each line, which are used to match the user's query with relevant content.

**Example Commands:**
  - **Ask a Query:** Type your question and the system will retrieve relevant content from vault.txt and provide a response.

  - **Quit the Script:** Type quit to exit the conversation loop.

**Configuration:**
  - **Ollama API:** The script interacts with an Ollama API running locally on http://localhost:11434/v1. Ensure that the API is running before executing the script.

  - **Embedding Model**: The script uses the mxbai-embed-large model for generating document embeddings by default.

This script provides a simple, yet powerful tool for querying local documents and generating responses using RAG techniques without modifying the user's original input.

# Step 8: Test Description for Querying Complex Patient Records with RAG

This test demonstrates how the local Retrieval-Augmented Generation (RAG) system retrieves nuanced information from a complex JSON file of patient records. In this case, the query involves retrieving the treatments for Michael Lee's anxiety and understanding how his anxiety might be related to his hypertension.

**Query Example:**

```
What treatments are being used for Michael Lee's anxiety, and how is his condition related to his hypertension?
```
Context Retrieved from the `sample_complex.json` File:

The system pulls the relevant context from the JSON file, which includes:

   - Michael Lee's medical history, with conditions like Hypertension and Anxiety.
   - Treatments:
      - For hypertension, he is on Losartan 50mg daily.
      - For anxiety, he undergoes Cognitive Behavioral Therapy (CBT) and takes Escitalopram 10mg daily.
   - Recent visits that link his increased anxiety to a hypertension follow-up visit.

**RAG System's Response:**
Based on the retrieved context, the RAG system generated the following detailed response:
```
Anxiety Treatment: Michael Lee's anxiety is being treated with CBT and Escitalopram 10mg daily, a treatment regimen that began after his diagnosis in 2016.
Relation to Hypertension: The system highlighted a recent visit on May 5th, 2023, where Michael reported increased anxiety during a hypertension follow-up, suggesting a link between his anxiety and blood pressure issues. It also noted improvement in anxiety symptoms on September 15th, 2023, following continued CBT sessions.
```

The system also suggested potential additional factors outside the given context that might impact both conditions:

   - Exercise: Michael engages in light physical activity (occasional walks), but increasing exercise could help manage both conditions.
   - Sleep: Since Michael has sleep apnea, improving sleep quality could positively affect both his anxiety and hypertension.
   - Stress Management: The system recommended incorporating stress-reduction techniques like meditation or breathing exercises to better manage his anxiety and its effects on blood pressure.

**Why This Test Demonstrates the Power of RAG:**

This test showcases how RAG goes beyond traditional SQL queries by:

   - Understanding the relationships between seemingly separate conditions (anxiety and hypertension).
   - Pulling detailed contextual information from multiple areas of the patient’s record (treatments, recent visits, and lifestyle factors).
   - Generating actionable insights about how to manage the patient’s overall health, which would be difficult to achieve with a simple keyword search or SQL query.

By using the RAG system, healthcare providers can gain a more comprehensive understanding of a patient’s health conditions, treatments, and how different factors are interconnected, allowing for more personalized and effective care.

### Test Output: Querying Patient Distress Using RAG with Integrated Systems

**Query:**
```text
Which patients have shown signs of distress in recent facial recognition scans or feedback, and do they require urgent intervention?
```
### Response:

Based on the provided context and patient data, I've identified two patients who have shown signs of distress:

#### **Patient 006 (Sarah Miller)**:
- Recent visit on **2023-09-10** for "Increase in depressive symptoms," with **Sertraline dosage** increased to 100mg daily.
- A **facial recognition scan** on **2023-09-10 at 15:35:00** detected sadness, and her **emergency contact** was alerted.

#### **Patient 007 (David Brown)**:
- Recent visit on **2023-08-22** related to a manic episode, with **Lithium dosage** adjusted to 1200mg daily.
- No facial recognition alert, but a **customer review analysis** flagged high anxiety in feedback on **2023-09-08**, recommending escalation.

### Additional Insights and Recommendations:
- **Patient 006 (Sarah Miller)** may need urgent intervention due to her worsening depressive symptoms and the triggered facial recognition alert. Addressing her lifestyle factors, such as improving sleep quality and increasing physical activity, could aid in managing her condition.
  
- **Patient 007 (David Brown)** should also be closely monitored, as his recent feedback indicated high levels of anxiety. Although no facial recognition alert was triggered, further support may be necessary.

### Urgent Intervention Recommendations:
1. **Sarah Miller**: Address the increasing depressive symptoms and facial recognition-triggered alert.
2. **David Brown**: Provide additional support based on his flagged customer review analysis and anxiety escalation.

Please consult with the patient's care team and relevant healthcare professionals to determine the best course of action.
