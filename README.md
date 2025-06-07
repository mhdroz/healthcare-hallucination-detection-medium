# Transparent and Trustworthy Healthcare RAG

This project provides a complete framework for building and evaluating a Retrieval-Augmented Generation (RAG) system for the healthcare domain. It features a web interface built with FastAPI and a sophisticated backend that incorporates multiple layers of safety checks to mitigate AI hallucinations and ensure the reliability of generated answers.

The primary goal is to create a transparent and trustworthy medical AI that not only provides answers but also shows the evidence supporting them and gives a clear confidence score based on a rigorous, multi-faceted analysis.

## ✨ Key Features

- **Interactive Web Interface**: A clean UI built with FastAPI, Bootstrap, and JavaScript to query the RAG system and view detailed results.
- **Comprehensive Safety Analysis**: Each answer is subjected to a suite of safety checks before being shown to the user.
- **Attribution & Groundedness**: Verifies that the AI's answer is strongly supported by the retrieved medical literature.
- **Response Consistency**: Checks for stability by running multiple generation attempts to see if the core meaning remains consistent.
- **Uncertainty Measurement**: Uses semantic entropy to detect if the model is "unsure" about its own answer, a key indicator of potential hallucination.
- **External Fact-Checking**: (Optional) Validates the generated answer against external knowledge sources.
- **Source Transparency**: Clearly displays the exact source chunks from the medical literature used to generate the answer.
- **Weak Sentence Detection**: Flags and displays specific sentences in the answer that have low attribution scores.
- **End-to-End Evaluation**: Integrated `ragas` evaluation to grade the system's performance on metrics like Faithfulness and Answer Relevancy.
- **Modular and Configurable**: Easily configure the LLM, embedding models, rerankers, and safety checks via a central `config.py` file.

## ⚙️ How It Works

The system follows a multi-stage process when a user submits a query:

1.  **Query & Retrieval**: The user's question is converted into a vector embedding. The system performs a semantic search over a pre-built index of medical documents (in this case, on pneumonia treatment guidelines) to find the most relevant text chunks.
2.  **Reranking**: A reranker model re-orders the retrieved chunks to place the most relevant ones at the top.
3.  **Synthesis & Generation**: The top-ranked chunks and the original question are passed to a Large Language Model (e.g., GPT-4) which generates a synthesized answer.
4.  **Comprehensive Safety Check**: This is the crucial step. Before returning the answer, the system performs the multi-layered safety analysis described above (attribution, consistency, entropy, etc.).
5.  **Response & Display**: The final answer, along with all the safety scores, interpretations, and source documents, is sent to the FastAPI backend and rendered in the user interface.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.9+
- An [OpenAI API Key](https://platform.openai.com/docs/quickstart)

### 2. Setup

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/healthcare-hallucination-detection.git
cd healthcare-hallucination-detection
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**
Create a file named `.env` in the root of the project directory and add your OpenAI API key:
```
OPENAI_API_KEY="sk-..."
```

**5. Download the data corpus:**
The RAG system relies on a corpus of medical documents. Run the download script:
```bash
python -m scripts.download_corpus
```
This will download the necessary files into the `data/raw/` directory.

**6. Build the Vector Index:**
Next, process the raw data and build the vector index that the RAG system will query.
```bash
python -m scripts.build_index
```
This script will create a LlamaIndex vector store in the `data/indices/` directory.

### 3. Running the Application

Once the setup is complete, you can start the web application.

```bash
python -m app.main
```

The server will start. You can access the web interface by navigating to:
[**http://localhost:8000**](http://localhost:8000)

The API documentation is also available at `http://localhost:8000/api/docs`.

## 📁 Project Structure

```
.
├── app/                  # FastAPI Web Application
│   ├── main.py           # App entrypoint, startup/shutdown logic
│   ├── routes.py         # API endpoint definitions
│   ├── models.py         # Pydantic models for API
│   ├── utils.py          # Helper functions for the web app
│   ├── static/           # CSS and JavaScript files
│   └── templates/        # Jinja2 HTML templates
│
├── scripts/              # Standalone scripts
│   ├── download_corpus.py # Downloads raw data
│   └── build_index.py    # Builds the LlamaIndex vector store
│
├── src/                  # Core backend logic
│   ├── rag/              # RAG pipeline implementation
│   ├── safety/           # Safety check implementations
│   └── corpus/           # Data loading and processing
│
├── data/                 # Project data
│   ├── raw/              # Raw downloaded documents
│   └── indices/          # Stored vector indices
│
├── config.py             # Central configuration file
├── requirements.txt      # Python dependencies
└── README.md             # This file
```