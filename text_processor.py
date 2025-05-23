import easyocr
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import label

class TextProcessor:
    def __init__(self):
        # Initialize with multiple languages for better accuracy
        self.reader = easyocr.Reader(['en'], gpu=False)
        
    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise image
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened

    def filter_signboard_region(self, image):
        # Consider top half of image instead of just top third
        height = image.shape[0]
        top_portion = image[0:int(height/2), :]
        return top_portion

    def find_text_regions(self, image):
        # Create binary mask of text regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find connected components (text regions)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        
        # Label connected regions
        labels, num = label(dilate)
        
        # Find regions with high text density
        regions = []
        for i in range(1, num + 1):
            mask = (labels == i).astype(np.uint8)
            if cv2.countNonZero(mask) > 100:  # Filter small regions
                x, y, w, h = cv2.boundingRect(mask)
                density = cv2.countNonZero(mask) / (w * h)
                if density > 0.3:  # Only keep dense text regions
                    regions.append((x, y, w, h))
        
        return regions

    def draw_ocr_results(self, image, results):
        output = image.copy()
        overlay = image.copy()
        
        for (box, text, conf) in results:
            points = np.array(box).astype(np.int32)
            # Draw filled polygon for text region
            cv2.fillPoly(overlay, [points], (0, 255, 0, 0.3))
            # Draw text boundary
            cv2.polylines(output, [points], True, (0, 255, 0), 2)
            # Add text and confidence
            cv2.putText(output, f"{text} ({conf:.2f})", 
                       tuple(points[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Blend overlay with original
        alpha = 0.3
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
        
        return output

    def extract_text(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return [], None
            
        # Find text-dense regions
        regions = self.find_text_regions(image)
        
        all_texts = []
        filtered_results = []
        
        for (x, y, w, h) in regions:
            # Extract region with padding
            pad = 10
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(image.shape[1], x+w+pad), min(image.shape[0], y+h+pad)
            roi = image[y1:y2, x1:x2]
            
            # Get OCR results for region
            results = self.reader.readtext(roi)
            
            # Process results
            region_text = []
            for r in results:
                if len(r) == 3:  # Valid detection
                    box, text, conf = r
                    if conf > 0.4:
                        # Adjust bounding box coordinates to original image
                        adjusted_box = np.array(box) + [x1, y1]
                        filtered_results.append((adjusted_box.tolist(), text, conf))
                        region_text.append(text)
            
            if region_text:
                all_texts.append(' '.join(region_text))
        
        # Create visualization
        visualized = self.draw_ocr_results(image, filtered_results)
        
        return all_texts, visualized

    def clean_text(self, text):
        # Remove unwanted characters
        text = ''.join(c if c.isalnum() or c.isspace() or c in '&,-' else ' ' for c in text)
        # Remove multiple spaces
        text = ' '.join(text.split())
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
