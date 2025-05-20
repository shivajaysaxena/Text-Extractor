import easyocr
import cv2
import numpy as np
from collections import Counter
import re

class TextProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return enhanced

    def filter_signboard_region(self, image):
        # Consider top half of image instead of just top third
        height = image.shape[0]
        top_portion = image[0:int(height/2), :]
        return top_portion

    def extract_text(self, image_path):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return ""

        # Preprocess image
        preprocessed = self.preprocess_image(image)
        signboard_region = self.filter_signboard_region(preprocessed)

        # Detect text in signboard region
        result = self.reader.readtext(signboard_region)

        # Filter and process detected text
        texts = []
        for (bbox, text, prob) in result:
            # Only include text with high confidence
            if prob > 0.5:
                # Clean the text
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    texts.append(cleaned_text)

        return ' '.join(texts)

    def clean_text(self, text):
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s&,-]', '', text)
        # Convert to title case for business names
        text = text.title()
        return text.strip()

    def find_common_word(self, text):
        if not text:
            return None
            
        # Normalize and split text
        words = text.lower().strip().split()
        
        # Business-related keywords to prioritize
        business_words = {
            'traders', 'store', 'shop', 'stores', 'mobile', 'general', 
            'electronics', 'services', 'recharge', 'sales', 'trading'
        }
        
        # Enhanced stop words list
        stop_words = {
            'the', 'and', 'for', 'ltd', 'limited', 'pvt', 'private',
            'cell', 'ph', 'phone', 'mob', 'mobile', 'all', 'available',
            'sim', 'cards', 'road', 'download'
        }
        
        # Filter words and numbers
        words = [
            word for word in words 
            if (len(word) > 2 and 
                word not in stop_words and 
                not any(char.isdigit() for char in word))
        ]
        
        if not words:
            return None
        
        # Prioritize business-related words
        business_related = [word for word in words if word in business_words]
        if business_related:
            return max(business_related, key=len)
        
        # If no business words found, use the longest non-stop word
        return max(words, key=len)
