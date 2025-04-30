import os
import argparse
from typing import List
from agents.llm_agent import LLMAgent
from agents.scraper_agent import ScraperAgent
# We'll add the VLM agent later

def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(description="Fine-tuning Dataset Annotation Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Initial prompt describing the images you want")
    parser.add_argument("--num_initial", type=int, default=5, help="Number of initial images to download")
    parser.add_argument("--num_final", type=int, default=100, help="Number of final images to download")
    parser.add_argument("--download_dir", type=str, default="downloads", help="Directory to save downloaded images")
    
    args = parser.parse_args()
    
    # Initialize agents
    llm_agent = LLMAgent()
    scraper_agent = ScraperAgent(download_dir=args.download_dir)
    
    # Step 1: Generate search prompts
    search_prompts = llm_agent.generate_search_prompts(args.prompt)
    print(f"Generated search prompts: {search_prompts}")
    
    # Step 2: Download initial images using the first prompt
    initial_prompt = search_prompts[0]
    initial_images = scraper_agent.search_and_download_initial(initial_prompt, args.num_initial)
    print(f"Downloaded {len(initial_images)} initial images")
    
    # Step 3: User feedback (simulated for now)
    # In a real application, you would show the images to the user and get feedback
    # For now, we'll assume all images are relevant
    relevant_images = initial_images
    irrelevant_images = []
    
    # Step 4: Refine the prompt based on feedback
    refined_prompt = llm_agent.refine_prompt_from_feedback(
        initial_prompt, relevant_images, irrelevant_images
    )
    print(f"Refined prompt: {refined_prompt}")
    
    # Step 5: Download bulk images with the refined prompt
    bulk_images = scraper_agent.search_and_download_bulk(refined_prompt, args.num_final)
    print(f"Downloaded {len(bulk_images)} images with the refined prompt")
    
    # The next steps would involve the VLM for verification and annotation

if __name__ == "__main__":
    main() 