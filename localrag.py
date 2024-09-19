import ollama
import argparse
import json
import numpy as np
from openai import OpenAI

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss  # Import FAISS for efficient similarity search

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to generate embeddings for the vault content using Ollama
def generate_embeddings(vault_content):
    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])
    return np.array(vault_embeddings)

# Store embeddings in FAISS index
def index_embeddings(embeddings):
    dimension = embeddings.shape[1]  # Dimension of the embedding
    index = faiss.IndexFlatL2(dimension)  # L2 distance (equivalent to cosine for normalized vectors)
    index.add(embeddings)  # Add the embeddings to the FAISS index
    return index

# Function to save FAISS index
def save_faiss_index(index, file_path="faiss_index.bin"):
    faiss.write_index(index, file_path)

# Function to load FAISS index
def load_faiss_index(file_path="faiss_index.bin"):
    if os.path.exists(file_path):
        return faiss.read_index(file_path)
    else:
        return None

# Function to save embeddings
def save_embeddings(embeddings, file_path="vault_embeddings.npy"):
    np.save(file_path, embeddings)

# Function to load embeddings
def load_embeddings(file_path="vault_embeddings.npy"):
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        return None

# Function to perform the search in the FAISS index
def search_index(index, query_embedding, top_k=3):
    query_embedding = np.array(query_embedding).reshape(1, -1)  # Reshape query embedding
    distances, indices = index.search(query_embedding, top_k)  # Perform FAISS search
    return distances, indices

# Function to get relevant context from the vault based on user input using FAISS
def get_relevant_context_faiss(rewritten_input, index, vault_embeddings, vault_content, top_k=3):
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    
    # Perform search in the FAISS index
    distances, top_indices = search_index(index, np.array(input_embedding), top_k)
    
    # Retrieve relevant context based on top-k results
    relevant_context = [vault_content[idx] for idx in top_indices[0]]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})

def ollama_chat(user_input, system_message, faiss_index, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context_faiss(rewritten_query, faiss_index, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3.1", help="Ollama model to use (default: llama3.1)")
args = parser.parse_args()

# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Try to load saved FAISS index and embeddings
print(NEON_GREEN + "Attempting to load saved FAISS index and embeddings..." + RESET_COLOR)
faiss_index = load_faiss_index("faiss_index.bin")
vault_embeddings = load_embeddings("vault_embeddings.npy")

# If index or embeddings do not exist, generate them
if faiss_index is None or vault_embeddings is None:
    print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
    vault_embeddings = generate_embeddings(vault_content)

    # Index the embeddings using FAISS
    print(NEON_GREEN + "Indexing embeddings using FAISS..." + RESET_COLOR)
    faiss_index = index_embeddings(vault_embeddings)

    # Save the FAISS index and embeddings for future use
    print(NEON_GREEN + "Saving FAISS index and embeddings..." + RESET_COLOR)
    save_faiss_index(faiss_index, "faiss_index.bin")
    save_embeddings(vault_embeddings, "vault_embeddings.npy")
else:
    print(NEON_GREEN + "Loaded FAISS index and embeddings from disk." + RESET_COLOR)

# Conversation loop
print("Starting conversation loop...")
conversation_history = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."

while True:
    user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break
    
    response = ollama_chat(user_input, system_message, faiss_index, vault_embeddings, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
