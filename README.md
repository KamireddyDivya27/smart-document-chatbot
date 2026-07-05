# 📄 Smart Document Chatbot

A chatbot that lets you upload any PDF and ask questions about it, powered by Retrieval-Augmented Generation (RAG).

**Live Demo:** https://smart-document-chatbot-igdkrgsfxigchsektvmgsq.streamlit.app/


<img width="2560" height="1326" alt="image" src="https://github.com/user-attachments/assets/83812d0a-def6-4a7d-bb4a-d01583f08234" />


## Overview

Smart Document Chatbot extracts text from uploaded PDFs, breaks it into chunks, converts those chunks into vector embeddings, and stores them in a FAISS vector index. When a user asks a question, the app retrieves the most relevant chunks and passes them to Google's Gemini model to generate a grounded answer — one that's based only on the document's actual content.

## Features

- Upload one or more PDF documents
- Automatic text extraction and chunking
- Semantic search over document content using FAISS
- Context-grounded question answering (no hallucinated answers — if the answer isn't in the document, the bot says so)
- Simple chat interface built with Streamlit

## Tech Stack

- **Frontend/App Framework:** Streamlit
- **PDF Processing:** PyPDF2
- **Text Splitting & Orchestration:** LangChain
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS
- **LLM:** Google Gemini (`gemini-2.5-flash`)

## How It Works

1. **Upload:** User uploads one or more PDF files via the sidebar
2. **Extract:** Text is extracted from all pages of the PDFs
3. **Chunk:** Text is split into overlapping chunks (1000 characters, 200 character overlap) for better context retrieval
4. **Embed & Store:** Chunks are converted to vector embeddings and saved locally in a FAISS index
5. **Query:** When a user asks a question, the top 4 most relevant chunks are retrieved
6. **Answer:** The retrieved context and the question are passed to Gemini, which generates an answer grounded strictly in the document content

## Running Locally

```bash
# Clone the repository
git clone https://github.com/KamireddyDivya27/smart-document-chatbot.git
cd smart-document-chatbot

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Add your Google Gemini API key to a .env file
echo GOOGLE_API_KEY=your_key_here > .env

# Run the app
streamlit run app.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | API key for Google Gemini (get one at [Google AI Studio](https://aistudio.google.com/app/apikey)) |

## Project Structure

```
smart-document-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## Author

**Divya Kamireddy**
- GitHub: [@KamireddyDivya27](https://github.com/KamireddyDivya27)
- Email: divyakamireddy27@gmail.com
