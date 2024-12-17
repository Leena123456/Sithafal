from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key

app = Flask(__name__)

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text_data = []
    for page in reader.pages:
        text_data.append(page.extract_text())
    return text_data

def chunk_text(text_list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = []
    for text in text_list:
        chunks.extend(text_splitter.split_text(text))
    return chunks

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    return embeddings

def store_embeddings(embeddings, chunks):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

def retrieve_relevant_chunks(query, index, model, stored_chunks, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [stored_chunks[idx] for idx in indices[0]]

def generate_answer(question, context):
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response['choices'][0]['text'].strip()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle the PDF file upload
        uploaded_file = request.files['pdf']
        file_path = f"uploads/{uploaded_file.filename}"
        uploaded_file.save(file_path)

        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(pdf_text)

        # Embed the chunks and store in FAISS index
        chunk_embeddings = embed_chunks(chunks)
        faiss_index, stored_chunks = store_embeddings(chunk_embeddings, chunks)

        return render_template('index.html', faiss_index=faiss_index, stored_chunks=stored_chunks)

    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['question']
    faiss_index = request.form['faiss_index']
    stored_chunks = request.form['stored_chunks']

    # Retrieve relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(query, faiss_index, embedding_model, stored_chunks)

    # Generate the answer using OpenAI based on the retrieved context
    context = "\n".join(relevant_chunks)
    answer = generate_answer(query, context)

    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
