import os
from transformers import AutoTokenizer, AutoModel
import torch
from PyPDF2 import PdfReader
from pinecone import Pinecone as PineconeClient

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")

# Load PDF documents
def load_docs(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return [{"page_content": text}]

# Load all files in a directory
def load_dir(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            documents.extend(load_docs(file_path))
    return documents

# Split documents into chunks
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    chunks = []
    for doc in documents:
        text = doc["page_content"]
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append({"page_content": chunk})
    return chunks

# Embed text using HuggingFace transformers
def embed_text(texts, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=False)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Upsert documents into Pinecone
def upsert_docs(file):
    # Check if the path is a file or a directory
    if os.path.isfile(file):
        document = load_docs(file)
    elif os.path.isdir(file):
        document = load_dir(file)

    chunk_docs = split_docs(document)
    
    texts = [doc["page_content"] for doc in chunk_docs]
    embeddings = embed_text(texts, EMBED_MODEL_NAME)
    
    # Initialize Pinecone with the new method
    pc = PineconeClient(
    api_key=PINECONE_API_KEY 
    )
    index = pc.Index(INDEX_NAME)
    
    # Prepare metadata for Pinecone upsert
    vectors = [(str(i), embedding.tolist(), {"text": chunk_docs[i]["page_content"]}) for i, embedding in enumerate(embeddings)]
    index.upsert(vectors)

    return index
