import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch 
from openai import AzureOpenAI
from doc_load import upsert_docs
from groq import Groq

# Load environment variables from .env file
load_dotenv()


# Retrieve API keys
# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME_GPT_35_TURBO = os.getenv("DEPLOYMENT_NAME_GPT_35_TURBO")
DEPLOYMENT_NAME_LLAMA_33_70B_VERSATILE = os.getenv("DEPLOYMENT_NAME_LLAMA_33_70B_VERSATILE")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

NUMBER_OF_RESULT = 3


# Initialize Pinecone with the new method
pc = PineconeClient(
    api_key=PINECONE_API_KEY 
)


# Ensure the index exists or create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Adjust dimension based on your use case
        metric='cosine',
        spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
    )


# Initialize Pinecone index
index = pc.Index(INDEX_NAME)

# Set up the Azure OpenAI client
# client = AzureOpenAI(
#     api_key=AZURE_OPENAI_API_KEY,
#     azure_endpoint=ENDPOINT,
#     #azure_ad_token_provider=token_provider,
#     api_version="2024-05-01-preview",
# )

client = Groq(api_key= GROQ_API_KEY)

# HuggingFace embedding model
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, use_fast=True, clean_up_tokenization_spaces=False)
model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Function to get embeddings from Azure OpenAI
# def get_embeddings(texts):
#     response = client.embeddings.create(
#         model=DEPLOYMENT_NAME, # 'text-embedding-ada-002'
#         input=texts
#     )
#     embeddings = [embedding['embedding'] for embedding in response['data']]
#     return np.array(embeddings)


# Function to generate a summary using Azure OpenAI
def generate_answer(user_query, retrieved_context):
    response = client.chat.completions.create(
        model= DEPLOYMENT_NAME_LLAMA_33_70B_VERSATILE,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are an intelligent AI assistant with access to specific context and a user query. Your task is to generate a thoughtful and accurate response based on the provided context. Use your intelligence to interpret and synthesize the information, but do not add any details not found in the context. If the context does not contain information relevant to the user’s query, respond with something like 'Sorry, I cannot answer the question which is Out of Scope'"

                    "### User Query:"
                    f"{user_query}\n\n"
                    
                    "### Context:"
                    f"{retrieved_context}\n\n"
                    
                    "Please generate a response that is insightful and directly derived from the context. Avoid introducing any information that is not present in the context. If the context does not provide sufficient information to answer the query, simply respond with something like 'Sorry, I cannot answer the question which is Out of Scope'"
                    
                    "### Response:"

                )
            }
        ],
        max_tokens=100,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )
    answer = response.choices[0].message.content.strip()
    return answer

# Function to query the model and Pinecone
def run_llm(query: str):
    query_embedding = embed_text(query)
    result = index.query(vector=query_embedding.tolist(), top_k=NUMBER_OF_RESULT, include_metadata=True)
    context = " ".join([match['metadata']['text'] for match in result['matches']])
    response = generate_answer(user_query=query, retrieved_context=context)
    return response

# Streamlit UI

st.title("Interactive Guide")
st.write("Ask a question and get relevant information.")

query = st.text_input("Enter your question:")
if st.button("Submit"):
    if query:
        response = run_llm(query)
        st.write("Answer:")
        st.write(response)
    else:
        st.error("Please enter a question.")

# Upload section
st.header("Upload Files")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    upsert_docs(file_path)
    st.success("File uploaded and processed successfully!")
