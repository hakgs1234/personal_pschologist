import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import pdfplumber

# CSS styling for a beautiful layout
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #4e79a7;
            text-align: center;
        }
        .about-section, .project-section, .team-section {
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .team-photo {
            display: flex;
            justify-content: center;
            margin-top: 10px;
        }
        .container {
            width: 90%;
            margin: auto;
        }
        footer {
            text-align: center;
            font-size: 0.9em;
            margin-top: 20px;
            color: #777;
        }
    </style>
""", unsafe_allow_html=True)

# Define page sections
def about_us():
    st.markdown("<div class='about-section'><h2>About Us</h2>", unsafe_allow_html=True)
    st.markdown("""
        We are a team dedicated to providing accessible mental health support through AI. 
        This application combines insights from recognized psychology resources to respond 
        thoughtfully and compassionately to users' mental health concerns.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

def project_details():
    st.markdown("<div class='project-section'><h2>Project Details</h2>", unsafe_allow_html=True)
    st.markdown("""
        This AI-driven platform is designed to assist individuals by providing support 
        and resources related to mental health. Based on input, it draws on books such as 
        the DSM-5 for diagnostic information and "Theories of Personality" for understanding 
        personality disorders, offering a thoughtful response to users.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

def team_details():
    st.markdown("<div class='team-section'><h2>Our Team</h2>", unsafe_allow_html=True)
    st.markdown("""
        Our team consists of AI enthusiasts, mental health advocates, and professionals 
        committed to enhancing the accessibility of mental health resources. We use the latest 
        in AI research to develop this support platform.
    """)
    st.image("team_photo.jpg", caption="Meet the Team", use_column_width=True)  # Replace with an actual team photo
    st.markdown("</div>", unsafe_allow_html=True)

# Load and extract text from multiple PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return ""
    return text

# Load PDFs
pdf_files = {
    "DSM-5": 'Diagnostic and statistical manual of mental disorders _ DSM-5 ( PDFDrive.com ).pdf',
    "Theories of Personality": 'b6c3v8_Theories_of_Personality_10.pdf'
}
all_texts = {}

# Extract text from each PDF and store in dictionary
for title, pdf_file in pdf_files.items():
    st.write(f"Extracting text from {title}...")
    text = extract_text_from_pdf(pdf_file)
    if text:
        all_texts[title] = text
    else:
        st.warning(f"No text found in {pdf_file}")

# Chunk text from PDFs
def chunk_text(text, chunk_size=300):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Initialize embeddings and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)

for title, text in all_texts.items():
    chunks = chunk_text(text)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    embeddings_np = embeddings.detach().cpu().numpy()
    index.add(embeddings_np)
    all_texts[title] = (chunks, embeddings_np)

# Set up the generation model
generator = pipeline("text-generation", model="gpt2")

# Select relevant book based on query
def choose_book(query):
    if "personality" in query.lower():
        return "Theories of Personality"
    else:
        return "DSM-5"

# Generate response based on query
def generate_response(query):
    try:
        book_title = choose_book(query)
        st.write(f"Using knowledge from: {book_title}")
        chunks, embeddings_np = all_texts[book_title]
        
        query_embedding = model.encode([query], convert_to_tensor=True).detach().cpu().numpy()
        k = 3
        _, retrieved_indices = index.search(query_embedding, k)
        
        if retrieved_indices is not None and len(retrieved_indices[0]) > 0:
            retrieved_chunks = [chunks[idx] for idx in retrieved_indices[0] if idx < len(chunks)]
        else:
            retrieved_chunks = ["I'm here to help. Let's work through this together."]
        
        context = " ".join(retrieved_chunks)
        prompt = f"User query: {query}\n\nRelevant context: {context}\n\nResponse:"
        response = generator(prompt, max_new_tokens=150, num_return_sequences=1)[0]["generated_text"]
        return response
    
    except Exception as e:
        st.error(f"An error occurred while generating response: {e}")
        return f"An error occurred: {e}"

# Streamlit Navigation and User Input
st.title("Personal Psychologist AI")

# Navigation
menu = ["Chat", "About Us", "Project Details", "Our Team"]
choice = st.sidebar.selectbox("Navigate", menu)

if choice == "About Us":
    about_us()
elif choice == "Project Details":
    project_details()
elif choice == "Our Team":
    team_details()
else:
    st.markdown("### AI Mental Health Support Chat")
    user_input = st.text_input("Enter your query:")
    if user_input:
        st.write("Processing your query...")
        response = generate_response(user_input)
        st.write(f"Response: {response}")

# Footer
st.markdown("<footer>© 2024 Personal Psychologist AI - All rights reserved</footer>", unsafe_allow_html=True)
