#Task 1: Chat with PDF using RAG (Retrieval-Augmented Generation) Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to enable conversational AI with a PDF document. Users can upload a PDF file, and the system extracts its content, chunks the text, and indexes it for efficient retrieval. When users ask a question, the pipeline retrieves the most relevant information from the PDF and generates a context-aware response using a language model like GPT.

### Key Features:
- **PDF Parsing**: Extracts and preprocesses text from uploaded PDFs.
- **Text Chunking**: Splits large PDF content into manageable chunks for better retrieval.
- **Vector Store**: Creates embeddings for efficient semantic search using libraries like FAISS or Chroma.
- **RAG Workflow**: Combines retrieved chunks with generative models to answer user queries.
- **Streamlit UI**: Provides an interactive and user-friendly interface for uploading PDFs and asking questions.

### Tech Stack:
- **Python**: Core language for implementing the pipeline.
- **LangChain**: For retrieval and language model integration.
- **OpenAI API**: To generate conversational responses.
- **Streamlit**: For creating the web interface.
- **PyPDF2**: For PDF parsing.
- **Chroma/FAISS**: For creating and querying vector stores.

### How It Works:
1. **PDF Upload**: The user uploads a PDF via the Streamlit interface.
2. **Document Processing**: The PDF text is extracted, chunked, and embedded.
3. **Question Answering**:
   - The user's query is converted into an embedding.
   - Relevant chunks are retrieved from the vector store.
   - The generative model combines retrieved chunks with the query to produce a response.

### Use Cases:
- Research Assistance
- Summarizing lengthy PDF documents
- Conversational Q&A on technical manuals, reports, or e-books

### Installation & Usage:
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Run the app using `streamlit run app.py`.
4. Upload a PDF and start asking questions!

![Initial page]([https://i.postimg.cc/XJJ9m5Q2/Screenshot-2024-07-15-140854.png](https://i.postimg.cc/13NZrKS2/Whats-App-Image-2024-12-17-at-08-49-58-0e20360d.jpg))

