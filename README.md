# Task 1: Chat with PDF using RAG (Retrieval-Augmented Generation) Pipeline

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

![Task1](https://i.postimg.cc/T3mnMZG9/Whats-App-Image-2024-12-17-at-08-49-58-0e20360d.jpg)

# Task 2: Chat with Website Using RAG Pipeline

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to enable interactive querying of structured and unstructured data from websites. The pipeline crawls, scrapes, processes website content, and provides context-rich, accurate responses using a selected Large Language Model (LLM).

## Functional Requirements

### 1. Data Ingestion
- **Input:** URLs or list of websites to crawl/scrape.
- **Process:**
  - Crawl and scrape content from target websites.
  - Extract key data fields, metadata, and textual content.
  - Segment content into smaller chunks for granularity.
  - Convert chunks into vector embeddings using a pre-trained embedding model.
  - Store embeddings in a vector database for efficient retrieval.

### 2. Query Handling
- **Input:** User's natural language query.
- **Process:**
  - Convert the user's query into vector embeddings using the same embedding model.
  - Perform a similarity search in the vector database to find relevant chunks.
  - Provide these chunks as context to the LLM for generating responses.

### 3. Response Generation
- **Input:** Retrieved data and the user query.
- **Process:**
  - Use the LLM with retrieval-augmented prompts to produce context-aware responses.
  - Ensure factuality by incorporating retrieved data directly into the response.

## Example Websites
- [University of Chicago](https://www.uchicago.edu/)
- [University of Washington](https://www.washington.edu/)
- [Stanford University](https://www.stanford.edu/)
- [University of North Dakota](https://und.edu/)

## How It Works
1. **Crawl and Scrape:** Extract content from provided URLs.
2. **Vectorize Data:** Convert data chunks to vector embeddings.
3. **Query:** Accept user queries and retrieve relevant data using similarity search.
4. **Generate Response:** Use LLM to generate detailed and factual answers.

## Technologies Used
- **Web Scraping:** `BeautifulSoup`, `Scrapy`
- **Embedding Model:** `OpenAI Embeddings` or `SentenceTransformers`
- **Vector Database:** `Chroma`, `Pinecone`, or `FAISS`
- **Large Language Model:** OpenAI GPT or similar
- **Framework:** `Streamlit` for the interactive UI

![Task2](https://i.postimg.cc/d1jSw5P7/Whats-App-Image-2024-12-17-at-08-49-40-ee2e1e8a.jpg)

