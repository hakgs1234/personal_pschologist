import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Retrieve Hugging Face token from Streamlit secrets
HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN")

# Check if the token is available and valid
if not HUGGINGFACE_TOKEN:
    st.error("Hugging Face token not found. Please set the HUGGINGFACE_TOKEN in Streamlit Secrets.")
    st.stop()

# Function to load model and tokenizer
def load_model(model_path):
    try:
        # Load tokenizer and model using Hugging Face token for gated models
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGINGFACE_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_auth_token=HUGGINGFACE_TOKEN)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path from Hugging Face
model_path = "Izza-shahzad-13/fine-tuned-flan-t5"  # Replace with actual model path

# Load tokenizer and model
tokenizer, model = load_model(model_path)
if model and tokenizer:
    model.to(device)
else:
    st.stop()  # Stop the app if model or tokenizer failed to load

# Function to generate response
def generate_response(input_text, book_contexts):
    # Add context from books to the input text
    context = " ".join(book_contexts)
    input_with_context = context + " " + input_text
    
    # Tokenize input
    inputs = tokenizer(input_with_context, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_length=500,
            num_beams=4,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            do_sample=True,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Function to get most relevant book context using FAISS similarity search
def get_relevant_context(user_input, book_embeddings, faiss_index):
    # Encode user input and perform similarity search
    input_embeddings = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_vector = model.encoder(input_embeddings['input_ids']).last_hidden_state.mean(dim=1).cpu().detach().numpy()
    
    # Search for the most relevant context using FAISS
    D, I = faiss_index.search(input_vector, k=3)  # Adjust 'k' based on how many contexts you want
    relevant_context = [book_embeddings[i] for i in I[0]]
    
    return relevant_context

# Example book embedding and FAISS setup (you will replace this with actual book data)
book_texts = ["b6c3v8_Theories_of_Personality_10.pdf", "Diagnostic and statistical manual of mental disorders _ DSM-5 ( PDFDrive.com ).pdf"]  # Replace with actual book texts
book_embeddings = [tokenizer.encode(text, return_tensors='pt').cpu().detach().numpy() for text in book_texts]
faiss_index = faiss.IndexFlatL2(len(book_embeddings[0]))  # Use FAISS index for similarity search
faiss_index.add(np.vstack(book_embeddings))  # Add the book embeddings to the FAISS index

# Streamlit app interface
st.title("FLAN-T5 Mental Health Counseling Assistant")
st.write("Type your thoughts or feelings, and let the model respond.")

# User input
user_input = st.text_area("How are you feeling today?", placeholder="Type here...")

# Generate response when input is provided
if user_input.strip():
    with st.spinner("Generating response..."):
        relevant_context = get_relevant_context(user_input, book_embeddings, faiss_index)
        response = generate_response(user_input, relevant_context)
    st.write("Model Response:", response)
else:
    st.info("Please enter your thoughts or feelings in the text area above.")
