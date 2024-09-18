import ollama
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tkinter as tk
from tkinter import filedialog

# Step 1: Document Loading
def load_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".json"):
        loader = JSONLoader(file_path)
    else:
        raise ValueError("Unsupported file format.")
    
    documents = loader.load()
    return documents

# Step 2: Generate Embeddings using Ollama
def generate_embeddings(documents):
    # Use Ollama to get embeddings for each document
    embeddings = []
    texts = []
    for doc in documents:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=doc.page_content)
        embeddings.append(response['embedding'])
        texts.append(doc.page_content)  # Store the original text for retrieval

    # Convert the embeddings into a numpy array
    embeddings_np = np.array(embeddings)
    
    # Create the FAISS index using embeddings and the associated documents' texts
    vectorstore = FAISS.from_embeddings(embeddings_np, texts)
    
    return vectorstore

# Step 3: Set up Memory for Conversational Retrieval Chain
memory = ConversationBufferMemory(memory_key="chat_history")

# Step 4: Combine Everything in Conversational Retrieval Chain
def setup_conversational_chain(vectorstore):
    # Use Ollama to handle queries locally with LLaMA
    def query_ollama(prompt):
        response = ollama.chat(model="llama3.1", prompt=prompt)
        return response['response']
    
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=query_ollama, 
        retriever=vectorstore.as_retriever(), 
        memory=memory
    )
    return conversational_chain

# Function to interact with the user
def ask_query(conversational_chain):
    while True:
        user_query = input("Enter your query (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break

        response = conversational_chain.run({"query": user_query})
        print(f"Response: {response}")

# File Upload via GUI
def upload_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("JSON Files", "*.json")])
    return file_path

# Main function to orchestrate the workflow
def main():
    print("Please select a file (PDF, TXT, or JSON) to upload:")
    file_path = upload_file()

    if not file_path:
        print("No file selected. Exiting...")
        return

    # Step 1: Load the document
    print("Loading document...")
    documents = load_documents(file_path)

    # Step 2: Generate Embeddings and Create Vector Store
    print("Generating embeddings and indexing the document...")
    vectorstore = generate_embeddings(documents)

    # Step 3: Set up the conversational retrieval chain
    print("Setting up the conversational chain with memory...")
    conversational_chain = setup_conversational_chain(vectorstore)

    # Step 4: Start asking user queries
    print("You can now ask questions about the document:")
    ask_query(conversational_chain)

if __name__ == "__main__":
    main()
