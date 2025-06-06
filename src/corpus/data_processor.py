"""
Data processing utilities for PubMed corpus.
"""
import json
import os
from typing import List, Dict, Optional
#from pathlib import Path
from app.utils import debug_print


class DataProcessor:
    """Processes and saves PubMed article data."""
    
    @staticmethod
    def save_articles_jsonl(articles: List[Dict], output_path: str) -> None:
        """
        Save articles to a JSONL file.
        
        Args:
            articles: List of article dictionaries
            output_path: Path to output JSONL file
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                json.dump(article, f, ensure_ascii=False)
                f.write('\n')
        
        debug_print(f"Saved {len(articles)} articles to {output_path}")
    
    @staticmethod
    def load_articles_jsonl(input_path: str) -> List[Dict]:
        """
        Load articles from a JSONL file.
        
        Args:
            input_path: Path to input JSONL file
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    article = json.loads(line)
                    articles.append(article)
        
        debug_print(f"Loaded {len(articles)} articles from {input_path}")
        return articles
    
    @staticmethod
    def validate_article(article: Dict) -> bool:
        """
        Validate that an article has required fields.
        
        Args:
            article: Article dictionary
            
        Returns:
            True if article is valid
        """
        required_fields = ["pmcid", "title", "abstract", "full_text"]
        
        for field in required_fields:
            if field not in article:
                return False
            if not article[field] or not article[field].strip():
                return False
        
        return True
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common artifacts
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.replace("\r", " ")
        
        return text.strip()
    
    @classmethod
    def clean_articles(cls, articles: List[Dict]) -> List[Dict]:
        """
        Clean and validate a list of articles.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of cleaned and validated articles
        """
        cleaned_articles = []
        
        for article in articles:
            # Skip invalid articles
            if not cls.validate_article(article):
                debug_print(f"Skipping invalid article: {article.get('pmcid', 'Unknown')}")
                continue
            
            # Clean text fields
            article["title"] = cls.clean_text(article["title"])
            article["abstract"] = cls.clean_text(article["abstract"])
            article["full_text"] = cls.clean_text(article["full_text"])
            
            cleaned_articles.append(article)
        
        debug_print(f"Cleaned {len(cleaned_articles)} valid articles out of {len(articles)}")
        return cleaned_articles
    
    @staticmethod
    def get_corpus_stats(articles: List[Dict]) -> Dict:
        """
        Get statistics about the corpus.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary with corpus statistics
        """
        if not articles:
            return {"total_articles": 0}
        
        stats = {
            "total_articles": len(articles),
            "avg_title_length": 0,
            "avg_abstract_length": 0,
            "avg_full_text_length": 0,
            "license_distribution": {},
            "year_distribution": {}
        }
        
        title_lengths = []
        abstract_lengths = []
        full_text_lengths = []
        
        for article in articles:
            # Text lengths
            title_lengths.append(len(article.get("title", "")))
            abstract_lengths.append(len(article.get("abstract", "")))
            full_text_lengths.append(len(article.get("full_text", "")))
            
            # License distribution
            license_type = article.get("license", "unknown")
            stats["license_distribution"][license_type] = stats["license_distribution"].get(license_type, 0) + 1
            
            # Year distribution
            pub_date = article.get("publication_date", "")
            year = pub_date.split("-")[0] if pub_date else "unknown"
            stats["year_distribution"][year] = stats["year_distribution"].get(year, 0) + 1
        
        # Calculate averages
        if title_lengths:
            stats["avg_title_length"] = sum(title_lengths) / len(title_lengths)
        if abstract_lengths:
            stats["avg_abstract_length"] = sum(abstract_lengths) / len(abstract_lengths)
        if full_text_lengths:
            stats["avg_full_text_length"] = sum(full_text_lengths) / len(full_text_lengths)
        
        return stats
    
    @staticmethod
    def filter_articles_by_date(articles: List[Dict], start_year: Optional[int] = None, 
                               end_year: Optional[int] = None) -> List[Dict]:
        """
        Filter articles by publication date.
        
        Args:
            articles: List of article dictionaries
            start_year: Minimum publication year (inclusive)
            end_year: Maximum publication year (inclusive)
            
        Returns:
            Filtered list of articles
        """
        filtered_articles = []
        
        for article in articles:
            pub_date = article.get("publication_date", "")
            if not pub_date:
                continue
            
            try:
                year = int(pub_date.split("-")[0])
                
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue
                
                filtered_articles.append(article)
                
            except (ValueError, IndexError):
                # Skip articles with invalid dates
                continue
        
        debug_print(f"Filtered to {len(filtered_articles)} articles "
              f"(from {start_year or 'any'} to {end_year or 'any'})")
        
        return filtered_articles