import easyocr
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import label
from difflib import SequenceMatcher
from collections import Counter

class TextProcessor:
    def __init__(self):
        # Initialize with multiple languages for better accuracy
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Define validation parameters
        self.min_confidence = 0.45
        self.min_text_length = 3
        self.max_text_length = 50
        self.valid_chars_ratio = 0.65
        self.text_cluster_distance = 30  # pixels

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

    def detect_signboards(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        
        # Edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Dilate edges to connect components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter potential signboard regions
        signboards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Filter small regions
                continue
                
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = w/h
            
            # Check if shape is suitable for a signboard
            if 1.2 < aspect_ratio < 6.0:  # Typical signboard ratios
                roi = image[y:y+h, x:x+w]
                # Check if ROI has enough contrast
                if self.has_text_features(roi):
                    signboards.append((x,y,w,h))
        
        return signboards

    def has_text_features(self, roi):
        # Convert to grayscale if not already
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
        # Calculate histogram features
        hist = cv2.calcHist([roi], [0], None, [256], [0,256])
        hist_norm = hist.ravel()/hist.sum()
        
        # Check contrast and distribution
        std_dev = np.std(roi)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
        
        return std_dev > 30 and entropy > 3.0

    def validate_text(self, text, conf):
        if conf < self.min_confidence:
            return False
            
        # Remove spaces for length check
        clean_text = text.strip()
        if len(clean_text) < self.min_text_length or len(clean_text) > self.max_text_length:
            return False
            
        # Check ratio of valid characters
        valid_chars = sum(1 for c in clean_text if c.isalnum() or c.isspace())
        if valid_chars / len(clean_text) < self.valid_chars_ratio:
            return False
            
        return True

    def extract_text(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return [], None
            
        # Get text using both approaches
        results = []
        
        # Get segmentation results
        signboards = self.detect_signboards(image)
        for (x,y,w,h) in signboards:
            roi = image[y:y+h, x:x+w]
            detections = self.reader.readtext(roi)
            for detection in detections:
                if len(detection) == 3:
                    box, text, conf = detection
                    if conf > self.min_confidence:
                        results.append(text)
        
        # Group results into text blocks
        text_groups = []
        if results:
            text_groups = [results]  # Single group for now, can be enhanced
            
        # Create visualization
        visualized = self.draw_results(image, text_groups)
        
        # Clean up text groups before returning
        text_groups = [[str(text) for text in group] for group in text_groups]
        
        return text_groups, visualized

    def get_segmentation_results(self, image):
        signboards = self.detect_signboards(image)
        results = []
        
        for (x,y,w,h) in signboards:
            roi = image[y:y+h, x:x+w]
            detections = self.reader.readtext(roi)
            
            for detection in detections:
                if len(detection) == 3:
                    box, text, conf = detection
                    if conf > self.min_confidence:
                        # Adjust coordinates to original image
                        adjusted_box = np.array(box) + [x, y]
                        results.append((adjusted_box.tolist(), text, conf, (x+w/2, y+h/2)))
        
        return results

    def get_clustering_results(self, image):
        # Find text-dense regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        
        # Get raw OCR results
        detections = self.reader.readtext(blur)
        results = []
        
        for detection in detections:
            if len(detection) == 3:
                box, text, conf = detection
                if conf > self.min_confidence:
                    center = np.mean(box, axis=0)
                    results.append((box, text, conf, tuple(center)))
        
        return results

    def merge_detection_results(self, seg_results, cluster_results):
        merged = []
        used_positions = set()
        
        # Add all segmentation results
        for result in seg_results:
            box, text, conf, center = result
            merged.append(result)
            used_positions.add(center)
        
        # Add clustering results that don't overlap
        for result in cluster_results:
            box, text, conf, center = result
            if not any(self.points_close(center, pos, self.text_cluster_distance) 
                      for pos in used_positions):
                merged.append(result)
                used_positions.add(center)
        
        return merged

    def group_text_by_proximity(self, results):
        if not results:
            return []
            
        # Sort by vertical position
        results.sort(key=lambda x: x[3][1])  # Sort by y-coordinate of center
        
        groups = []
        current_group = [results[0][1]]  # Start with first text
        last_y = results[0][3][1]
        
        for result in results[1:]:
            text = result[1]
            current_y = result[3][1]
            
            if current_y - last_y > self.text_cluster_distance:
                groups.append(current_group)
                current_group = []
            
            current_group.append(text)
            last_y = current_y
            
        if current_group:
            groups.append(current_group)
            
        return groups

    def points_close(self, p1, p2, threshold):
        return np.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2)) < threshold

    def draw_results(self, image, text_groups):
        output = image.copy()
        overlay = image.copy()
        
        # Draw each text group
        y_offset = 30  # Starting y position for text display
        for group in text_groups:
            # Create a text block for this group
            text_block = ' '.join(group)
            
            # Add text to image
            cv2.putText(output, text_block, 
                       (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            y_offset += 30  # Move down for next text block
        
        # Add semi-transparent overlay
        alpha = 0.3
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
        
        return output

    def clean_text(self, text):
        # Remove unwanted characters
        text = ''.join(c if c.isalnum() or c.isspace() or c in '&,-' else ' ' for c in text)
        # Remove multiple spaces
        text = ' '.join(text.split())
        return text.strip()

    def find_common_word(self, text):
        if not text:
            return None
            
        # Handle list input
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text)
        
        # Normalize and split text
        try:
            words = text.lower().strip().split()
        except AttributeError:
            print(f"Error processing text: {text}")
            return None
        
        # Business-related keywords to prioritize
        business_words = {
            'traders', 'store', 'shop', 'stores', 'mobile', 'general', 
            'electronics', 'services', 'stationar', 'gift', 'collection'
        }
        
        # Enhanced stop words list
        stop_words = {
            'the', 'and', 'for', 'ltd', 'limited', 'pvt', 'private',
            'cell', 'ph', 'phone', 'mob', 'etc', 'brand', 'well'
        }
        
        # Filter words
        words = [
            word for word in words 
            if (len(word) > 2 and 
                word not in stop_words and 
                not word.isdigit() and
                not any(char.isdigit() for char in word))
        ]
        
        if not words:
            return None
        
        # Prioritize business-related words
        business_related = [word for word in words if word in business_words]
        if business_related:
            # Get most frequent business word
            word_counts = Counter(business_related)
            return max(word_counts.items(), key=lambda x: (x[1], len(x[0])))[0]
        
        # If no business words found, use most frequent word
        word_counts = Counter(words)
        return max(word_counts.items(), key=lambda x: (x[1], len(x[0])))[0]

    def organize_text(self, detections):
        # Organize text into categories
        text_info = {
            'business_name': [],
            'products': [],
            'contact': [],
            'address': [],
            'other': []
        }
        
        for text in detections:
            cleaned_text = str(text).strip()
            if any(char.isdigit() for char in cleaned_text):
                if len(cleaned_text) > 8:  # Likely phone/mobile
                    text_info['contact'].append(cleaned_text)
                else:
                    text_info['other'].append(cleaned_text)
            elif any(word in cleaned_text.lower() for word in ['road', 'street', 'lane', 'nagar']):
                text_info['address'].append(cleaned_text)
            elif any(word in cleaned_text.lower() for word in ['shop', 'store', 'traders', 'collection']):
                text_info['business_name'].append(cleaned_text)
            elif any(word in cleaned_text.lower() for word in ['stationary', 'gift', 'general', 'mobile', 'services']):
                text_info['products'].append(cleaned_text)
            else:
                text_info['other'].append(cleaned_text)
        
        # Format organized text
        formatted_text = []
        for category, texts in text_info.items():
            if texts:
                if category == 'business_name':
                    formatted_text.insert(0, f"Business Name: {' '.join(texts)}")
                elif category == 'products':
                    formatted_text.append(f"Products/Services: {', '.join(texts)}")
                elif category == 'contact':
                    formatted_text.append(f"Contact: {', '.join(texts)}")
                elif category == 'address':
                    formatted_text.append(f"Address: {' '.join(texts)}")
        
        return '\n'.join(formatted_text)

    def extract_text(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return [], None
            
        # Get text using both approaches
        results = []
        
        # Get segmentation results
        signboards = self.detect_signboards(image)
        for (x,y,w,h) in signboards:
            roi = image[y:y+h, x:x+w]
            detections = self.reader.readtext(roi)
            for detection in detections:
                if len(detection) == 3:
                    box, text, conf = detection
                    if conf > self.min_confidence:
                        results.append(text)
        
        # Group results into text blocks
        text_groups = []
        if results:
            text_groups = [results]  # Single group for now, can be enhanced
            
        # Create visualization
        visualized = self.draw_results(image, text_groups)
        
        # Clean up text groups before returning
        text_groups = [[str(text) for text in group] for group in text_groups]
        
        # Process results through organizer
        organized_texts = []
        for group in text_groups:
            organized_text = self.organize_text(group)
            if organized_text:
                organized_texts.append(organized_text)
        
        return organized_texts, visualized
