#!/usr/bin/env python3
"""
CLI script to download medical corpus from PubMed Central.

Usage:
    python scripts/download_corpus.py --query "infectious disease" --max-results 500
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from corpus import PubMedDownloader, DataProcessor
from dotenv import load_dotenv


def main():
    """Main function to download and process PubMed corpus."""
    parser = argparse.ArgumentParser(description="Download medical corpus from PubMed Central")
    
    parser.add_argument(
        "--query", 
        type=str, 
        default="infectious disease OR pneumonia OR sepsis",
        help="Search query for PubMed Central"
    )
    parser.add_argument(
        "--max-results", 
        type=int, 
        default=500,
        help="Maximum number of articles to download"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=50,
        help="Number of articles to fetch per batch"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="pmc_articles.jsonl",
        help="Output filename"
    )
    parser.add_argument(
        "--allowed-licenses",
        nargs="+",
        default=["cc-by", "cc-by-sa", "cc0"],
        help="Allowed Creative Commons licenses"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between API calls in seconds"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API credentials
    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("EMAIL")
    
    if not email:
        print("Warning: EMAIL not set in environment. This is recommended for NCBI API usage.")
    
    if not api_key:
        print("Warning: NCBI_API_KEY not set. Using without API key (lower rate limits).")
    
    # Initialize downloader
    downloader = PubMedDownloader(api_key=api_key, email=email)
    
    # Download articles
    print(f"Starting download with query: '{args.query}'")
    print(f"Max results: {args.max_results}")
    print(f"Allowed licenses: {args.allowed_licenses}")
    
    articles = downloader.download_articles(
        query=args.query,
        max_results=args.max_results,
        batch_size=args.batch_size,
        delay=args.delay,
        allowed_licenses=set(args.allowed_licenses)
    )
    
    if not articles:
        print("No articles downloaded. Exiting.")
        return
    
    # Clean and validate articles
    print("\nCleaning and validating articles...")
    cleaned_articles = DataProcessor.clean_articles(articles)
    
    # Save to file
    output_path = os.path.join(args.output_dir, args.output_file)
    DataProcessor.save_articles_jsonl(cleaned_articles, output_path)
    
    # Print statistics
    print("\n" + "="*50)
    print("CORPUS STATISTICS")
    print("="*50)
    
    stats = DataProcessor.get_corpus_stats(cleaned_articles)
    
    print(f"Total articles: {stats['total_articles']}")
    print(f"Average title length: {stats['avg_title_length']:.1f} characters")
    print(f"Average abstract length: {stats['avg_abstract_length']:.1f} characters") 
    print(f"Average full text length: {stats['avg_full_text_length']:.1f} characters")
    
    print(f"\nLicense distribution:")
    for license_type, count in stats['license_distribution'].items():
        print(f"  {license_type}: {count}")
    
    print(f"\nYear distribution (top 5):")
    year_items = sorted(stats['year_distribution'].items(), key=lambda x: x[1], reverse=True)
    for year, count in year_items[:5]:
        print(f"  {year}: {count}")
    
    print(f"\nCorpus saved to: {output_path}")
    print("Download complete!")


if __name__ == "__main__":
    main()