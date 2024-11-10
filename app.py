import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import pdfplumber  # Alternative PDF extraction library

# Load and extract text from multiple PDFs using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        return f"Error extracting text: {e}"
    return text

# Specify the PDF files you want to use
pdf_files = ['Diagnostic and statistical manual of mental disorders _ DSM-5 ( PDFDrive.com ).pdf',
             'b6c3v8_Theories_of_Personality_10.pdf']

all_text = ""

# Extract text from each PDF and combine
for pdf_file in pdf_files:
    all_text += extract_text_from_pdf(pdf_file) + "\n"  # Separate texts by new lines for clarity

# Split the combined text into chunks
def chunk_text(text, chunk_size=300):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks = chunk_text(all_text)

# Embed and index text chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, convert_to_tensor=True)

# Convert embeddings to numpy for FAISS
embeddings_np = embeddings.detach().cpu().numpy()

# Initialize FAISS index
embedding_dim = embeddings_np.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_np)

# Set up the generation model
generator = pipeline("text-generation", model="gpt2")

# Generate response based on query
def generate_response(query):
    try:
        # Step 1: Embed the query
        query_embedding = model.encode([query], convert_to_tensor=True).detach().cpu().numpy()

        # Step 2: Search FAISS for top-k similar chunks
        k = 3
        _, retrieved_indices = index.search(query_embedding, k)
        
        # Step 3: Check if retrieved_indices has results
        if retrieved_indices is not None and len(retrieved_indices[0]) > 0:
            retrieved_chunks = [chunks[idx] for idx in retrieved_indices[0] if idx < len(chunks)]
        else:
            retrieved_chunks = ["I'm here to help. Let's work through this together."]
        
        # Step 4: Combine retrieved chunks and pass to generator
        context = " ".join(retrieved_chunks)
        prompt = f"User is feeling overwhelmed and needs support. Here’s some information that might help: {context}\n\nUser query: {query}\n\nSupportive response:"
        
        # Generate the response using GPT-2 with max_new_tokens
        response = generator(prompt, max_new_tokens=150, num_return_sequences=1)[0]["generated_text"]
        return response
    
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit Interface
st.title("Personal Psychologist AI")
st.markdown("This AI provides mental health support based on books. Please enter your query below.")
user_input = st.text_input("Enter your query:")

if user_input:
    response = generate_response(user_input)
    st.write(response)
