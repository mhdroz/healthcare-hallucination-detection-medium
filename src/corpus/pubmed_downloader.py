"""
PubMed Central article downloader with license filtering.
"""
import requests
import xml.etree.ElementTree as ET
import time
import os
from typing import List, Dict, Optional, Set
from .license_detector import LicenseDetector


class PubMedDownloader:
    """Downloads articles from PubMed Central with license filtering."""
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """
        Initialize the PubMed downloader.
        
        Args:
            api_key: NCBI API key (optional but recommended for higher rate limits)
            email: Email address for NCBI requests
        """
        self.api_key = api_key
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.license_detector = LicenseDetector()
    
    def search_pmc(self, query: str, max_results: int = 100) -> List[str]:
        """
        Search PubMed Central for articles matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of PMC IDs
        """
        search_url = f"{self.base_url}esearch.fcgi"
        search_params = {
            "db": "pmc",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        
        if self.api_key:
            search_params["api_key"] = self.api_key
        if self.email:
            search_params["email"] = self.email
        
        print(f"Searching PMC for: {query}")
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        
        result = response.json()
        ids = result["esearchresult"]["idlist"]
        print(f"Found {len(ids)} articles")
        
        return ids
    
    def fetch_articles_batch(self, pmc_ids: List[str]) -> ET.Element:
        """
        Fetch a batch of articles from PMC.
        
        Args:
            pmc_ids: List of PMC IDs to fetch
            
        Returns:
            XML root element containing the articles
        """
        fetch_url = f"{self.base_url}efetch.fcgi"
        fetch_params = {
            "db": "pmc",
            "id": ",".join(pmc_ids),
            "retmode": "xml"
        }
        
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        if self.email:
            fetch_params["email"] = self.email
        
        response = requests.get(fetch_url, params=fetch_params)
        response.raise_for_status()
        
        return ET.fromstring(response.content)
    
    def extract_article_data(self, article: ET.Element, pmc_id: str) -> Optional[Dict]:
        """
        Extract article data from XML element.
        
        Args:
            article: XML article element
            pmc_id: PMC ID for the article
            
        Returns:
            Dictionary with article data or None if extraction fails
        """
        try:
            article_data = {
                "pmcid": f"PMC{pmc_id}",
                "title": "",
                "abstract": "",
                "full_text": "",
                "publication_date": "",
                "authors": [],
                "license": "other"
            }
            
            # Extract title
            title_elem = article.find(".//article-title")
            if title_elem is not None:
                article_data["title"] = "".join(title_elem.itertext()).strip()
            
            # Extract abstract
            abstract_parts = article.findall(".//abstract//p")
            if abstract_parts:
                article_data["abstract"] = " ".join(
                    "".join(p.itertext()).strip() for p in abstract_parts
                )
            
            # Extract publication date
            pub_date = article.find(".//pub-date")
            if pub_date is not None:
                year = pub_date.find("year")
                month = pub_date.find("month")
                day = pub_date.find("day")
                date_parts = []
                if year is not None:
                    date_parts.append(year.text)
                if month is not None:
                    date_parts.append(month.text)
                if day is not None:
                    date_parts.append(day.text)
                article_data["publication_date"] = "-".join(date_parts)
            
            # Extract authors
            author_elems = article.findall(".//contrib[@contrib-type='author']")
            for author_elem in author_elems:
                surname = author_elem.find(".//surname")
                given_names = author_elem.find(".//given-names")
                author = {}
                if surname is not None:
                    author["surname"] = surname.text
                if given_names is not None:
                    author["given_names"] = given_names.text
                if author:
                    article_data["authors"].append(author)
            
            # Extract full text
            body = article.find(".//body")
            if body is not None:
                paragraphs = body.findall(".//p")
                article_data["full_text"] = " ".join(
                    "".join(p.itertext()).strip() for p in paragraphs
                )
            
            # Extract license
            license_elem = article.find(".//license")
            article_data["license"] = self.license_detector.detect_cc_license(license_elem)
            
            return article_data
            
        except Exception as e:
            print(f"Error extracting article data for PMC{pmc_id}: {e}")
            return None
    
    def download_articles(
        self,
        query: str,
        max_results: int = 100,
        batch_size: int = 20,
        delay: float = 0.2,
        allowed_licenses: Set[str] = {"cc-by", "cc-by-sa", "cc0"}
    ) -> List[Dict]:
        """
        Download and process articles from PubMed Central.
        
        Args:
            query: Search query
            max_results: Maximum number of articles to download
            batch_size: Number of articles to fetch per batch
            delay: Delay between API calls in seconds
            allowed_licenses: Set of allowed license types
            
        Returns:
            List of article dictionaries
        """
        # Search for articles
        pmc_ids = self.search_pmc(query, max_results)
        
        if not pmc_ids:
            print("No articles found")
            return []
        
        articles = []
        skipped = 0
        
        # Process in batches
        for i in range(0, len(pmc_ids), batch_size):
            batch_ids = pmc_ids[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1} ({len(batch_ids)} articles)...")
            
            # Add delay between batches
            if i > 0:
                time.sleep(delay)
            
            try:
                # Fetch batch
                root = self.fetch_articles_batch(batch_ids)
                
                # Process each article in the batch
                for idx, article in enumerate(root.findall(".//article")):
                    if idx >= len(batch_ids):
                        break
                        
                    pmc_id = batch_ids[idx]
                    article_data = self.extract_article_data(article, pmc_id)
                    
                    if article_data is None:
                        skipped += 1
                        continue
                    
                    # Check license
                    if not self.license_detector.is_allowed_license(
                        article_data["license"], allowed_licenses
                    ):
                        print(f"Skipping PMC{pmc_id} due to license: {article_data['license']}")
                        skipped += 1
                        continue
                    
                    articles.append(article_data)
                    print(f"Added PMC{pmc_id} (license: {article_data['license']})")
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"\nDownload complete:")
        print(f"- Downloaded: {len(articles)} articles")
        print(f"- Skipped: {skipped} articles")
        
        return articles