from typing import List, Dict, Any
import os
import requests
from urllib.parse import quote_plus
import time
import random
from ..scrapers.google_scraper import GoogleImageScraper

class ScraperAgent:
    """
    Agent responsible for coordinating image scraping from various sources.
    """
    
    def __init__(self, download_dir: str = "downloads"):
        """
        Initialize the scraper agent.
        
        Args:
            download_dir: Directory to save downloaded images
        """
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        
        # Initialize scrapers
        self.google_scraper = GoogleImageScraper()
        # Add more scrapers as needed
        
    def search_and_download_initial(self, search_query: str, num_images: int = 5) -> List[str]:
        """
        Search for and download an initial batch of images.
        
        Args:
            search_query: Query to search for
            num_images: Number of images to download
            
        Returns:
            List of paths to downloaded images
        """
        # Use Google scraper for initial images
        image_urls = self.google_scraper.search(search_query, num_images)
        
        # Download the images
        image_paths = []
        for i, url in enumerate(image_urls):
            try:
                # Create a unique filename
                ext = url.split('.')[-1].split('?')[0]
                if len(ext) > 4:  # If extension is too long, default to jpg
                    ext = 'jpg'
                
                filename = f"{search_query.replace(' ', '_')}_{i}.{ext}"
                filepath = os.path.join(self.download_dir, filename)
                
                # Download the image
                response = requests.get(url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    
                    image_paths.append(filepath)
                
                # Add a small delay to avoid rate limiting
                time.sleep(random.uniform(0.5, 1.5))
                
            except Exception as e:
                print(f"Error downloading image {url}: {e}")
        
        return image_paths
    
    def search_and_download_bulk(self, search_query: str, num_images: int = 100) -> List[str]:
        """
        Search for and download a larger batch of images after prompt refinement.
        
        Args:
            search_query: Refined query to search for
            num_images: Number of images to download
            
        Returns:
            List of paths to downloaded images
        """
        # Similar to initial download but with more images
        # In a real implementation, you might want to use multiple scrapers
        # and implement more sophisticated error handling and deduplication
        
        # For now, we'll reuse the same method
        return self.search_and_download_initial(search_query, num_images) 