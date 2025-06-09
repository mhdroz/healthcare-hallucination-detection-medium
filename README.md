# Trustworthy Healthcare AI Demo

**Complete RAG system with real-time safety evaluation and transparent interface**

This project demonstrates how to build a modular, interpretable AI system for healthcare using:
- Retrieval-Augmented Generation (RAG)
- Multi-layered hallucination detection
- External fact-checking (via Semantic Scholar)
- A transparent, real-time dashboard built with Streamlit

It is the companion codebase to the [4-part blog series](https://pub.towardsai.net/hallucinations-in-healthcare-llms-why-they-happen-and-how-to-prevent-them-614d845242f4) on building trustworthy LLM systems in healthcare.

---

## Features

- RAG pipeline with PubMed-based corpus
- Safety checks: Attribution, consistency, entropy, and external validation
- Visual dashboard: Confidence scores, source chunks, and radar charts
- Modular FastAPI backend + Streamlit frontend
- Built-in RAGAS evaluation with interpretation

---

## Quickstart

1. **Clone the repo and install dependencies**

```bash
git clone https://github.com/yourusername/healthcare-ai-dashboard.git
cd healthcare-ai-dashboard
pip install -r requirements.txt
```
2. **Set up your .env file**

Copy the .env_template to setup your .env with your own API keys

3. **Update config**
Edit config.py to set your preferred models (OpenAI, reranker, embedding model, etc.)

4. **Run the backend (FastAPI)**
```
uvicorn app.main:app --reload --port 8000
```

5. **Run the dashboard (Streamlit)**
```
streamlit run streamlit.py
```
Then open your browser at http://localhost:8501

## Project Structure
.
├── streamlit.py          # Main frontend app (Streamlit)
├── config.py             # Central configuration and paths
├── requirements.txt      # Python dependencies
├── test_app.ipynb        # Notebook to test the app and build the Index
├── README.md
├── .env_template         # Template for .env
│
├── src/                  # Core backend logic
│   ├── rag/              # Retrieval-augmented generation modules
│   ├── safety/           # Safety checks and fact validation
│   ├── corpus/           # PubMed data loading and processing
│   └── utils/            # Shared utilities
│
├── data/
│   ├── processed/        # Processed articles 
│   └── indices/          # Vector indices
│
├── notebooks/            # Jupyter notebooks from previous blog posts

## Blog Series
Part 1: [Why Hallucinations Matter in Healthcare](https://pub.towardsai.net/hallucinations-in-healthcare-llms-why-they-happen-and-how-to-prevent-them-614d845242f4)
Part 2: [Building a RAG System](https://medium.com/towards-artificial-intelligence/how-to-build-a-rag-system-for-healthcare-minimize-hallucinations-in-llm-outputs-0b8ea4a4eaae)
Part 3: [Detecting Hallucinations](https://medium.com/towards-artificial-intelligence/detecting-hallucinations-in-healthcare-ai-99aa67e55bb7)
Part 4: [Building a Transparent Interface]()

## Contributing
Feel free to fork, play, and adapt the system to your own domain.

## Questions or feedback?
Let’s connect on [LinkedIn](https://www.linkedin.com/in/marie-humbert-droz/)
