"""
Document processing utilities for RAG system.
Based on blog post 2 code.
"""
import json
from typing import List, Dict
from llama_index.core import Document


class DocumentProcessor:
    """Processes medical articles for RAG indexing."""
    
    @staticmethod
    def load_medical_articles(file_path: str) -> List[Dict]:
        """Load medical articles from a JSONL file."""
        articles = []
        with open(file_path, 'r') as f:
            for line in f:
                article = json.loads(line)
                articles.append(article)
        return articles
    
    @staticmethod
    def process_articles(articles: List[Dict]) -> List[Document]:
        """Process articles into LlamaIndex documents."""
        documents = []
        
        for article in articles:
            # Combine title, abstract and full text (as shown in blog post)
            full_content = f"Title: {article['title']}\n\nAbstract: {article['abstract']}\n\nFull Text: {article['full_text']}"
            
            # Create metadata
            metadata = {
                "pmcid": article['pmcid'],
                "title": article['title'],
                "publication_date": article['publication_date'],
                "source": "PubMed Central"
            }
            
            # Create LlamaIndex document
            doc = Document(
                text=full_content,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents