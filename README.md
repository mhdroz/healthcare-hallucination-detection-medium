# folder stricture:

healthcare-rag-system/
│
├── README.md
├── requirements.txt
├── .env.example
├── config.py
│
├── data/                          # Data storage
│   ├── raw/                       # Raw downloaded papers
│   ├── processed/                 # Processed JSONL files
│   └── indices/                   # Vector store indices
│
├── src/                           # Core application code
│   ├── __init__.py
│   │
│   ├── corpus/                    # Post 1: PubMed corpus generation
│   │   ├── __init__.py
│   │   ├── pubmed_downloader.py   # Download papers from PubMed Central
│   │   ├── license_detector.py    # Creative Commons license detection
│   │   └── data_processor.py      # Process and clean downloaded papers
│   │
│   ├── rag/                       # Post 2: RAG system
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Load and process documents
│   │   ├── indexer.py            # Create embeddings and vector index
│   │   ├── retriever.py          # Query engine and retrieval
│   │   └── chunking.py           # Different chunking strategies
│   │
│   ├── safety/                    # Post 3: Safety checks
│   │   ├── __init__.py
│   │   ├── attribution.py        # Source attribution scoring
│   │   ├── consistency.py        # Consistency checking
│   │   ├── entropy.py            # Semantic entropy measurement
│   │   ├── multi_stage.py        # Multi-stage retrieval
│   │   └── safety_checker.py     # Comprehensive safety assessment
│   │
│   ├── fact_check/                # Post 4: External fact-checking
│   │   ├── __init__.py
│   │   ├── external_sources.py   # Interface to external medical DBs
│   │   ├── validators.py         # Cross-reference with authorities
│   │   └── fact_checker.py       # Main fact-checking orchestrator
│   │
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       ├── embeddings.py         # Embedding model utilities
│       ├── llm_utils.py          # LLM initialization and helpers
│       └── text_processing.py    # Text cleaning and processing
│
├── web_app/                       # Post 4: Web interface
│   ├── __init__.py
│   ├── app.py                    # Main Flask/FastAPI application
│   ├── routes.py                 # API endpoints
│   ├── templates/                # HTML templates
│   │   ├── index.html
│   │   └── results.html
│   ├── static/                   # CSS, JS, images
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── models.py                 # Data models for web app
│
├── scripts/                       # Standalone scripts
│   ├── download_corpus.py        # CLI script to download corpus
│   ├── build_index.py            # CLI script to build RAG index
│   ├── evaluate_system.py        # CLI script to run evaluations
│   └── run_safety_tests.py       # CLI script to test safety modules
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_corpus/
│   ├── test_rag/
│   ├── test_safety/
│   └── test_fact_check/
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_corpus_exploration.ipynb
│   ├── 02_rag_evaluation.ipynb
│   ├── 03_safety_analysis.ipynb
│   └── 04_interface_demo.ipynb
│
└── docs/                          # Documentation
    ├── setup.md
    ├── api.md
    └── deployment.md