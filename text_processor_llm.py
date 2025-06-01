import google.generativeai as genai
import numpy as np
import cv2
from PIL import Image
import torch
import re
import io
import asyncio

class TextProcessorLLM:
    def __init__(self, api_key):
        print("[Init] Initializing LLM Text Processor...")
        
        # Configure Gemini with 1.5 Flash
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.2,      # Balanced between accuracy and flexibility
            "candidate_count": 1,    # Single focused response
            "max_output_tokens": 64, # Short responses for shop names
            "stop_sequences": [":", ";", ".", "\n"]  # Clean text boundaries
        }
        
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',  # Latest Flash model
            generation_config=generation_config
        )

        # Keep core corrections and patterns
        self.name_corrections = {
            'EHAGAWAN': 'BHAGAWAN',
            'BAGWAN': 'BHAGWAN',
            'MOB': None,
            'CELL': None
        }

        self.shop_keywords = {
            'bakery': {'bakery', 'baker', 'bread', 'cake'},
            'store': {'store', 'general store', 'provision'},
            'shop': {'shop', 'mart', 'market', 'emporium'},
            'names': {'bhagwan', 'krishna', 'ram', 'noor'}
        }

        # LLM prompts
        self.extraction_prompt = """
        Extract shop name and type from this signboard image. Follow these rules:
        1. Ignore phone numbers, contact details
        2. Remove noise like special characters
        3. Fix common misspellings
        4. Use proper capitalization
        5. Format as: "SHOP_NAME SHOP_TYPE"
        """

        self.verification_prompt = """
        Verify and correct this shop text:
        "{text}"
        Rules:
        1. Fix spelling errors
        2. Remove irrelevant words
        3. Keep proper business terms
        4. Return in format: "SHOP_NAME SHOP_TYPE"
        """

        # Initialize single event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def extract_text_sync(self, image_path):
        """Synchronous wrapper for extract_text"""
        try:
            return self.loop.run_until_complete(self.extract_text(image_path))
        except RuntimeError:
            # If loop is closed, create new one
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            return self.loop.run_until_complete(self.extract_text(image_path))
        finally:
            # Don't close the loop, just reset it
            self.loop._closed = False

    async def extract_text(self, image_path):
        try:
            # Open and convert image to bytes
            image = Image.open(image_path)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Generate content with Gemini
            response = await self.model.generate_content_async(
                [
                    self.extraction_prompt,
                    {"mime_type": "image/png", "data": img_bytes}
                ],
                stream=False
            )
            
            # Process response - updated for new Gemini response structure
            if response and hasattr(response, 'text'):
                extracted_text = response.text.strip()
            else:
                extracted_text = "No text detected"
            
            # Structure and return results - await the coroutine
            final_text = await self.structure_text(extracted_text)
            return [final_text], image, self.get_business_type(final_text)

        except Exception as e:
            print(f"[Extract] Error: {str(e)}")
            return ["Error processing image"], image, "shop"

    async def structure_text(self, text):  # Make method async
        """Structure text with LLM verification"""
        try:
            # Clean input
            text = re.sub(r'[^A-Za-z\s]', ' ', text)
            words = text.upper().split()
            
            # First pass: Apply corrections
            cleaned = []
            for word in words:
                if word in self.name_corrections:
                    if self.name_corrections[word]:
                        cleaned.append(self.name_corrections[word])
                elif len(word) >= 3:
                    cleaned.append(word)

            # Get LLM verification - await the coroutine
            verified_text = await self._verify_with_llm(' '.join(cleaned))
            
            # Split and process the verified text
            if verified_text:
                final_words = []
                for word in verified_text.split():
                    if word not in final_words:
                        final_words.append(word)
                return ' '.join(final_words)
            return "Unknown"

        except Exception as e:
            print(f"[Structure] Error: {str(e)}")
            return "Unknown"

    async def _verify_with_llm(self, text):
        """Verify text using Gemini"""
        try:
            response = await self.model.generate_content_async(
                self.verification_prompt.format(text=text),
                stream=False
            )
            return response.text.strip() if hasattr(response, 'text') else text
        except:
            return text

    def get_business_type(self, text):
        """Determine business type from verified text"""
        text_lower = text.lower()
        
        # Check keywords in order of priority
        if any(kw in text_lower for kw in self.shop_keywords['bakery']):
            return 'bakery'
        elif any(kw in text_lower for kw in self.shop_keywords['store']):
            return 'store'
        elif any(kw in text_lower for kw in self.shop_keywords['shop']):
            return 'shop'
        
        return 'shop'  # Default type
