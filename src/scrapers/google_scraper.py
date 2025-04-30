import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
from typing import List
import re
from urllib.parse import quote_plus

class GoogleImageScraper:
    """
    Scraper for Google Images.
    """
    
    def __init__(self):
        """
        Initialize the Google Images scraper.
        """
        # Set up Chrome options for headless browsing
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        
    def search(self, query: str, num_images: int = 10) -> List[str]:
        """
        Search Google Images for the given query.
        
        Args:
            query: Search query
            num_images: Number of image URLs to return
            
        Returns:
            List of image URLs
        """
        # Encode the query for URL
        encoded_query = quote_plus(query)
        search_url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"
        
        # Initialize the browser
        driver = webdriver.Chrome(options=self.chrome_options)
        
        try:
            # Load the page
            driver.get(search_url)
            
            # Wait for images to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img.rg_i"))
            )
            
            # Scroll down to load more images if needed
            if num_images > 20:
                for _ in range(num_images // 20):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
            
            # Extract image URLs
            image_urls = []
            elements = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
            
            for element in elements[:num_images]:
                # Try to get the image URL
                if element.get_attribute("src"):
                    image_urls.append(element.get_attribute("src"))
                elif element.get_attribute("data-src"):
                    image_urls.append(element.get_attribute("data-src"))
                
                # Break if we have enough images
                if len(image_urls) >= num_images:
                    break
            
            return image_urls
            
        finally:
            # Close the browser
            driver.quit() 