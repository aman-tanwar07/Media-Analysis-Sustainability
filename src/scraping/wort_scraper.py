"""
wort_scraper.py

Scraper for Luxemburger Wort articles.
Collects article URLs and publication dates for further analysis.

Usage:
    python src/scraping/wort_scraper.py

Output:
    data/raw/wort_articles.csv
"""

import requests
import gzip
from io import BytesIO
from datetime import datetime
from lxml import etree
from bs4 import BeautifulSoup
import re
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import logging
import pandas as pd
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE = "https://www.wort.lu"
START = datetime(2015, 1, 1)
END = datetime(2025, 6, 30)
HEADERS = {"User-Agent": "WortPublicScraper/1.0"}

# Compile regex patterns for your keywords
KEYWORD_PATTERNS = [
    re.compile(r"nachhaltig.*\sbauen", re.IGNORECASE),
    re.compile(r"nachhaltig.*\sGebäude", re.IGNORECASE),
]

# Configuration for async operations
MAX_CONCURRENT_REQUESTS = 10  # Adjust based on server capacity
REQUEST_DELAY = 0.1  # Reduced delay for async operations

class OptimizedScraper:
    def __init__(self):
        self.session = None
        self.results = []
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, limit_per_host=MAX_CONCURRENT_REQUESTS)
        self.session = aiohttp.ClientSession(
            headers=HEADERS,
            timeout=timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_bytes_async(self, url: str) -> bytes:
        """Async version of fetch_bytes with better error handling"""
        async with self.session.get(url) as response:
            if str(response.url).startswith("https://login.mediahuis.com"):
                raise ValueError("Redirected to login — protected content")
            response.raise_for_status()
            return await response.read()
    
    def fetch_bytes_sync(self, url: str) -> bytes:
        """Synchronous version for sitemap fetching"""
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        if resp.url.startswith("https://login.mediahuis.com"):
            raise ValueError("Redirected to login — protected content")
        return resp.content
    
    def get_sitemap_urls(self) -> List[str]:
        """Get sitemap URLs - kept synchronous as it's a single request"""
        logger.info("Fetching sitemap index...")
        xml = self.fetch_bytes_sync(f"{BASE}/sitemap.xml")
        root = etree.fromstring(xml)
        urls = [elem.text for elem in root.findall(".//{*}loc")]
        return [u for u in urls if "article" in u]
    
    def parse_article_sitemap(self, sitemap_url: str) -> List[Tuple[str, Optional[str]]]:
        """Parse article sitemap - optimized with better error handling"""
        try:
            gz_data = self.fetch_bytes_sync(sitemap_url)
            with gzip.GzipFile(fileobj=BytesIO(gz_data)) as f:
                xml = f.read()
            
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(xml, parser=parser)
            
            articles = []
            for url_elem in root.findall(".//{*}url"):
                loc_elem = url_elem.find("{*}loc")
                if loc_elem is None:
                    continue
                
                loc = loc_elem.text
                lm_elem = url_elem.find("{*}lastmod")
                lastmod = lm_elem.text if lm_elem is not None else None
                articles.append((loc, lastmod))
            
            return articles
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {e}")
            return []
    
    def filter_candidates(self, article_sitemaps: List[str]) -> List[Tuple[str, datetime]]:
        """Filter articles by date range using ThreadPoolExecutor"""
        candidates = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Process sitemaps in parallel
            sitemap_results = list(executor.map(self.parse_article_sitemap, article_sitemaps))
        
        for articles in sitemap_results:
            for loc, lastmod in articles:
                if lastmod:
                    try:
                        # More robust date parsing
                        date_str = lastmod.split("T")[0]
                        dt = datetime.fromisoformat(date_str)
                        if START <= dt <= END:
                            candidates.append((loc, dt))
                    except ValueError:
                        continue
        
        logger.info(f"Total candidate articles: {len(candidates)}")
        return sorted(candidates, key=lambda x: x[1])
    
    def matches_keywords(self, text: str) -> bool:
        """Check if text matches any keyword patterns"""
        return any(pattern.search(text) for pattern in KEYWORD_PATTERNS)
    
    async def process_article(self, url: str, dt: datetime, semaphore: asyncio.Semaphore) -> Optional[dict]:
        """Process single article with semaphore for rate limiting"""
        async with semaphore:
            try:
                content = await self.fetch_bytes_async(url)
                
                # Use lxml parser for better performance
                soup = BeautifulSoup(content, "lxml")
                text = soup.get_text()
                
                if self.matches_keywords(text):
                    title = soup.title.string.strip() if soup.title else "(no title)"
                    result = {
                        'date': dt.date(),
                        'title': title,
                        'url': url
                    }
                    logger.info(f"MATCH: {dt.date()} | {title}")
                    return result
                    
            except ValueError:
                logger.info(f"SKIP: Protected content: {url}")
            except Exception as e:
                logger.error(f"ERROR processing {url}: {e}")
            
            # Small delay to be polite
            await asyncio.sleep(REQUEST_DELAY)
            return None
    
    async def process_articles_batch(self, candidates: List[Tuple[str, datetime]]) -> List[dict]:
        """Process articles in batches with controlled concurrency"""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Create tasks for all articles
        tasks = [
            self.process_article(url, dt, semaphore) 
            for url, dt in candidates
        ]
        
        # Process in batches to avoid overwhelming the server
        batch_size = 50  # Process 50 articles at a time
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
            
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, dict):  # Valid result
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
            
            # Brief pause between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(1)
        
        return results
    
    def save_results_to_file(self, results: List[dict], format_type: str = 'both') -> str:
        """Save results to CSV and/or Excel file"""
        if not results:
            logger.info("No results to save")
            return ""
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_range = f"{START.strftime('%Y%m%d')}_to_{END.strftime('%Y%m%d')}"
        base_filename = f"wort_articles_{date_range}_{timestamp}"
        
        # Save to CSV
        if format_type in ['csv', 'both']:
            csv_filename = f"{base_filename}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            logger.info(f"Results saved to CSV: {csv_filename}")
            print(f"✅ CSV saved: {csv_filename}")
        
        # Save to Excel
        if format_type in ['excel', 'xlsx', 'both']:
            excel_filename = f"{base_filename}.xlsx"
            
            # Create Excel writer with some formatting
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Articles', index=False)
                
                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Articles']
                
                # Adjust column widths
                worksheet.column_dimensions['A'].width = 12  # Date
                worksheet.column_dimensions['B'].width = 80  # Title
                worksheet.column_dimensions['C'].width = 60  # URL
                
                # Make URLs clickable
                from openpyxl.styles import Font
                blue_font = Font(color='0000FF', underline='single')
                for row in range(2, len(df) + 2):  # Skip header
                    cell = worksheet[f'C{row}']
                    cell.font = blue_font
                    cell.hyperlink = cell.value
            
            logger.info(f"Results saved to Excel: {excel_filename}")
            print(f"✅ Excel saved: {excel_filename}")
        
        # Print summary
        print(f"📊 Summary: {len(results)} articles saved")
        
        return base_filename

    async def run_scraper(self, save_format: str = 'both') -> List[dict]:
        """Main scraper function"""
        start_time = time.time()
        
        # Get sitemaps (synchronous)
        article_sitemaps = self.get_sitemap_urls()
        logger.info(f"Found {len(article_sitemaps)} article sitemaps")
        
        # Filter candidates (parallel but synchronous)
        candidates = self.filter_candidates(article_sitemaps)
        
        if not candidates:
            logger.info("No articles found in date range")
            return []
        
        # Process articles (async)
        results = await self.process_articles_batch(candidates)
        
        end_time = time.time()
        logger.info(f"Scraping completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Found {len(results)} matching articles out of {len(candidates)} candidates")
        
        # Save results to file
        if results:
            self.save_results_to_file(results, save_format)
        
        return results

async def main(save_format: str = 'both'):
    """Main entry point"""
    async with OptimizedScraper() as scraper:
        results = await scraper.run_scraper(save_format)
        
        # Print results
        print("\n" + "="*80)
        print("MATCHING ARTICLES:")
        print("="*80)
        for result in results:
            print(f"{result['date']} | {result['title']}")
            print(f"  {result['url']}\n")
        
        return results

def run_scraper_sync(save_format: str = 'both'):
    """Synchronous wrapper for environments without event loop"""
    try:
        return asyncio.run(main(save_format))
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        return []
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise

async def run_scraper_async(save_format: str = 'both'):
    """Async wrapper for environments with existing event loop (Jupyter, etc.)"""
    try:
        return await main(save_format)
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        return []
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise

def detect_and_run(save_format: str = 'both'):
    """Detect environment and run appropriate version"""
    try:
        # Try to get current event loop
        loop = asyncio.get_running_loop()
        print("Running in async environment (Jupyter/IPython)")
        # If we're here, we have a running loop - use nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(main(save_format))
        except ImportError:
            print("nest_asyncio not installed. Please install it: pip install nest-asyncio")
            print("Alternatively, use: await run_scraper_async('both')")
            return None
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        print("Running in standard Python environment")
        return run_scraper_sync(save_format)

if __name__ == "__main__":
    # Install required packages:
    # pip install aiohttp lxml beautifulsoup4 requests nest-asyncio pandas openpyxl
    
    # You can specify the save format:
    # 'csv' - CSV only
    # 'excel' or 'xlsx' - Excel only  
    # 'both' - Both CSV and Excel (default)
    
    detect_and_run('both')
