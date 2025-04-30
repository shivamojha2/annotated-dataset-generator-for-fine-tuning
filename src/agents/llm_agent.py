import os
from typing import List, Dict, Any
import openai

class LLMAgent:
    """
    Agent that uses LLM to generate and refine image search prompts.
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Initialize the LLM agent.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: Model to use for generation
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        self.model = model
    
    def generate_search_prompts(self, user_prompt: str, num_prompts: int = 3) -> List[str]:
        """
        Generate search prompts based on user input.
        
        Args:
            user_prompt: User's description of the images they want
            num_prompts: Number of search prompts to generate
            
        Returns:
            List of search prompts
        """
        system_message = """
        You are an expert at generating effective image search queries.
        Given a description of the type of images a user wants to collect for ML training,
        generate specific, diverse search queries that will yield high-quality, relevant images.
        Focus on descriptive terms that will help find varied examples of the concept.
        """
        
        user_message = f"""
        I need to collect images for training a machine learning model.
        Description of images I need: {user_prompt}
        
        Please generate {num_prompts} effective search queries that will help me find diverse,
        high-quality images matching this description. Each query should be on a new line.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
        )
        
        # Extract and clean the generated prompts
        content = response.choices[0].message.content
        prompts = [p.strip() for p in content.strip().split('\n') if p.strip()]
        
        return prompts[:num_prompts]  # Ensure we return exactly the requested number
    
    def refine_prompt_from_feedback(self, 
                                   original_prompt: str, 
                                   relevant_images: List[str],
                                   irrelevant_images: List[str],
                                   vlm_feedback: str = None) -> str:
        """
        Refine the search prompt based on user feedback on initial images.
        
        Args:
            original_prompt: Original search prompt
            relevant_images: List of paths to images marked as relevant
            irrelevant_images: List of paths to images marked as irrelevant
            vlm_feedback: Optional feedback from VLM about the images
            
        Returns:
            Refined search prompt
        """
        # This would be implemented with a call to the LLM
        # For now, we'll return a placeholder
        system_message = """
        You are an expert at refining image search queries based on feedback.
        Your goal is to create a more precise query that will find images similar to the ones
        marked as relevant, while avoiding characteristics of irrelevant images.
        """
        
        user_message = f"""
        Original search prompt: "{original_prompt}"
        
        The user has reviewed some initial images and provided feedback:
        - Relevant images: {len(relevant_images)}
        - Irrelevant images: {len(irrelevant_images)}
        
        VLM analysis of the relevant images: {vlm_feedback or "Not provided"}
        
        Please refine the search prompt to find more images like the relevant ones
        and fewer images like the irrelevant ones.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,
        )
        
        refined_prompt = response.choices[0].message.content.strip()
        return refined_prompt
