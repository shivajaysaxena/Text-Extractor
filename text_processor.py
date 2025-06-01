import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from sklearn.cluster import DBSCAN
from rapidfuzz import fuzz, process
from collections import Counter
import re

class TextProcessor:
    def __init__(self):
        print("[Init] Initializing TextProcessor...")
        # Initialize parameters
        self.min_confidence = 0.05
        self.easyocr_conf = 0.1
        
        # Update OCR params for better detection
        self.ocr_params = {
            'paragraph': False,  # Process as individual words
            'batch_size': 8,
            'min_size': 10,     # Detect smaller text
            'text_threshold': 0.05,  # More sensitive text detection
            'low_text': 0.3,    # Better for dark text
            'contrast_ths': 0.05,# More sensitive to contrast
            'width_ths': 0.3,   # Allow wider text
            'height_ths': 0.3,  # Allow taller text
            'mag_ratio': 2.0    # Larger magnification
        }
        
        # Initialize EasyOCR with basic settings
        self.easyocr_reader = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory='./models',
            download_enabled=True,
            recog_network='english_g2'
        )

        self.priority_patterns = {
            'bakery': [r'(?i)[a-z\s]*b[a@]k[e3]r[yi]e?s?'],
            'store': [r'(?i)(gen[e3]r[a@]l)?\s*st[o0]r[e3]s?'],
            'shop': [r'(?i)sh[o0]ps?'],
            'gift': [r'(?i)g[i1]fts?\s*sh[o0]p']
        }
        
        self.text_cleanup = {
            'numbers': [r'\d+', ''],
            'special': [r'[^\w\s&-]', ''],
            'spaces': [r'\s+', ' ']
        }
        
        # Enhanced business categorization with priorities
        self.business_categories = {
            'bakery': 10,  # Prioritize bakery
            'general store': 9,
            'gift shop': 8,
            'stationery': 8,
            'mobile': 7,
            'general': 6,
            'store': 5,
            'shop': 4
        }
        
        # Priority words for shop types
        self.priority_words = {
            'bakery': {'bakery', 'bake', 'fresh', 'bread', 'cake', 'sweet', 'confectionery'},
            'stationery': {'stationery', 'book', 'pen', 'paper', 'gift'},
            'general': {'general', 'store', 'shop', 'mart', 'emporium', 'traders'},
            'mobile': {'mobile', 'phone', 'cell', 'recharge', 'accessories'}
        }
        
        # Add specialized bakery patterns
        self.text_patterns = {
            'shop_name': [
                r'(?i)[a-z]*\s*b[a@]k[e3]ry',  # Any word followed by bakery
                r'(?i)b[a@]k[e3]r[sy]?\s*[a-z]*',  # Bakery followed by any word
                r'(?i)n[o0]{2}r\s*b[a@]k[e3]ry',  # Match NOOR BAKERY with variations
                r'(?i)s[h#]r[e3]{2}\s*[a@]rt',  # Match SHREE ART
                r'(?i)g[i1]ft\s*sh[o0]p'  # Match GIFT SHOP
                r'(?i)st[a@]t[i1][o0]n[e3]ry'  # Match STATIONERY variations
            ],
            'business_type': [
                r'(?i)g[e3]n[e3]r[a@]l\s*st[o0]r[e3]s?',  # Match GENERAL STORE(S)
                r'(?i)st[a@]t[i1][o0]n[e3]ry\s*sh[o0]p',  # Match STATIONERY SHOP
                r'(?i)b[o0]{2}k\s*sh[o0]p',  # Match BOOK SHOP
                r'(?i)b[a@]k[e3]ry'  # Match BAKERY
            ]
        }
        self.min_area_ratio = 0.01  # Reduced for smaller signboards
        self.min_signboard_width = 100  # Reduced for smaller signs
        self.max_signboard_height = 300  # Increased for taller signs
        self.min_text_height = 12  # Further reduced
        self.cluster_distance = 40  # Increased for better line grouping
        self.edge_density_threshold = 0.1  # Lower threshold for edge density
        self.min_area_ratio = 0.01  # Reduced for smaller signboards
        self.min_text_area = 50  # Minimum area for text regions
        self.min_text_width = 20  # Minimum width for text regions
        self.min_text_height = 10  # Minimum height for text regions
        self.max_text_width = 500  # Maximum width for text regions
        self.max_text_height = 200  # Maximum height for text regions
        self.max_text_regions = 10  # Limit number of text regions
        self.min_text_ratio = 0.1  # Minimum ratio of text area to signboard area
        self.min_text_confidence = 0.3  # Minimum confidence for text detection
        
        # Additional parameters for text processing
        self.ignored_words = {'the', 'and', 'for', 'in', 'at', 'of', 'to', 'a', 'an'}
        self.max_word_repeats = 2  # Limit repeated words
        self.min_word_length = 3

        self.char_sequences = {
            'common': ['th', 'er', 'on', 'an', 'en', 'es', 'ing', 'ion'],
            'shop': ['shop', 'store', 'mart', 'center', 'point']
        }
        self.min_word_conf = 0.2  # Increased threshold

        # Add text correction parameters
        self.word_similarity_threshold = 0.7
        self.word_distance_threshold = 0.2
        self.common_words = {
            'brothers', 'traders', 'center', 'point', 'mart', 
            'store', 'shop', 'sales', 'service'
        }

        # Add language and visualization parameters
        self.lang_detector = re.compile(r'[a-zA-Z]')
        self.min_english_ratio = 0.5
        self.draw_boxes = True
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.box_color = (0, 255, 0)  # Green
        self.text_color = (255, 0, 0)  # Blue

        # Add context-specific word groups
        self.word_groups = {
            'business_prefix': {'sri', 'shree', 'shri', 'new', 'the'},
            'business_suffix': {'traders', 'trading', 'store', 'stores', 'shop', 'mart', 'centre', 'center'},
            'common_names': {'bhagwan', 'bhagawan', 'krishna', 'ram', 'shiva'},
            'business_types': {'mobile', 'phones', 'grocery', 'general', 'gift', 'bakery'}
        }

        # Enhanced mobile shop specific patterns
        self.shop_patterns = {
            'mobile': {
                'prefixes': {'zs', 'ss', 'ms', 'rs'},
                'core_terms': {'mobile', 'phone', 'cell', 'tel'},
                'suffixes': {'point', 'shop', 'store', 'traders', 'centre'},
                'services': {'repair', 'service', 'accessories', 'recharge'}
            }
        }
        
        # Enhanced corrections dictionary
        self.word_corrections = {
            'PH': 'PHONE',
            'MOB': 'MOBILE',
            'MOBL': 'MOBILE',
            'MOBIL': 'MOBILE',
            'PHON': 'PHONE',
            'FONE': 'PHONE',
            'TEL': 'TELEPHONE',
            'TRDRS': 'TRADERS',
            'TRADRS': 'TRADERS',
            'CENTR': 'CENTRE',
            'CTR': 'CENTRE',
            'ACCESSOR': 'ACCESSORIES'
        }

        # Add confidence weights for different detection methods
        self.confidence_weights = {
            'easyocr': 1.2,
            'tesseract': 1.0,
            'positional': 1.1
        }

        # Enhanced segmentation parameters for bakery signs
        self.segment_params = {
            'min_area': 100,      # Reduced for smaller text
            'max_area': 20000,    # Reduced to avoid over-segmentation
            'min_ratio': 0.15,    # More permissive ratio
            'max_ratio': 12.0,    # Allow longer text
            'padding': 8,         # Increased padding
            'cluster_eps': 35,    # More precise clustering
            'cluster_min_samples': 2,  # Require at least 2 samples
            'overlap_threshold': 0.3,  # Allow more overlap
            'height_similarity': 0.7,  # Height similarity threshold
            'vertical_gap': 0.5,   # Maximum vertical gap ratio
            'horizontal_gap': 1.5  # Maximum horizontal gap ratio
        }

        # Text clustering with multi-scale approach
        self.text_cluster_params = {
            'distance_threshold': 0.4,    # Stricter distance
            'confidence_weight': 0.5,     # Balanced confidence
            'position_weight': 0.5,       # Equal position weight
            'min_cluster_size': 2,        # Require pairs
            'scales': [0.8, 1.0, 1.2],   # Multi-scale detection
            'merge_threshold': 0.6,       # Merge similar detections
            'line_height_ratio': 1.5,     # Line height similarity
            'word_gap_ratio': 3.0        # Maximum word gap
        }

        # Add bakery-specific parameters
        self.bakery_params = {
            'keywords': ['bakery', 'bake', 'bread', 'cake', 'pastry', 'sweet'],
            'min_keyword_conf': 0.3,   # Lower threshold for keywords
            'name_patterns': [
                r'(?i)n[o0]{2}r',     # NOOR variations
                r'(?i)gr[e3]{2}n',    # GREEN variations
                r'(?i)b[e3]rg'        # BERG variations
            ],
            'layout_weights': {
                'centered': 1.2,       # Boost centered text
                'stacked': 1.1,        # Boost stacked layout
                'large_text': 1.3      # Boost larger text
            }
        }

        # BERT spell checking
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.spell_checker = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        
        # Add focused corrections dictionary for business text
        self.business_corrections = {
            'KISHNA': 'KRISHNA',
            'STATNEKY': 'STATIONERY',
            'STATIONRY': 'STATIONERY',
            'CEMRE': 'CENTRE',
            'CEMTRE': 'CENTRE',
            'CTR': 'CENTRE',
            'STNTIONEKY': 'STATIONERY',
            'STATINERY': 'STATIONERY'
        }
        
        # Update minimum thresholds
        self.min_word_confidence = 0.4  # Balanced confidence threshold
        self.min_text_length = 3      # Minimum word length
        self.max_corrections = 2       # Limit corrections per word
        
        # Business name patterns
        self.name_patterns = {
            'phone': r'(?:cell|tel|phone|:|\+)?(?:\d[\d-,\s]*\d)',
            'noise': r'[0-9%*]+\s*[0-9%*]+',
            'symbols': r'[^\w\s-]'
        }

        # Add common business text corrections
        self.common_corrections = {
            'KISHNA': 'KRISHNA',
            'STATNEKY': 'STATIONERY',
            'STATIONERY': 'STATIONERY',
            'CEMRE': 'CENTRE',
            'CEMTRE': 'CENTRE',
            'USHNA': 'KRISHNA',
            'STATINERY': 'STATIONERY',
            'STATIONRY': 'STATIONERY',
            'STNTIONEKY': 'STATIONERY'
        }

        # Simplified corrections dictionary
        self.text_corrections = {
            'KISHNA': 'KRISHNA',
            'STATNEKY': 'STATIONERY',
            'CEMRE': 'CENTRE',
            'CEMTRE': 'CENTRE',
            'STNTIONEKY': 'STATIONERY',
            'USHNA': 'KRISHNA',
            'LISHNA': 'KRISHNA'
        }

        # Add business keywords for text processing
        self.business_keywords = {
            'stationery': {'stationery', 'stationers', 'paper', 'book', 'books'},
            'gift': {'gift', 'gifts', 'card', 'cards'},
            'centre': {'centre', 'center', 'central'},
            'shop': {'shop', 'store', 'mart', 'emporium'}
        }

        # Common name variations
        self.name_corrections = {
            'EHAGAWAN': 'BHAGAWAN',
            'BAGWAN': 'BHAGWAN',
            'BAGAWAN': 'BHAGWAN',
            'MOB': None,  # Ignore mobile-related prefixes
            'MOBL': None,
            'CELL': None,
            'TEL': None
        }

        # Update shop keywords with more variations
        self.shop_keywords = {
            'bakery': {'bakery', 'baker', 'bake', 'bread', 'cake', 'sweet', 'confectionery'},
            'store': {'store', 'stores', 'general store', 'general stores', 'provision'},
            'shop': {'shop', 'mart', 'market', 'emporium', 'centre', 'center'},
            'names': {'bhagwan', 'bhagawan', 'shree', 'sri', 'shri', 'krishna', 'ram', 'noor'}
        }

        # Add name patterns
        self.name_patterns = {
            'prefixes': {'shri', 'shree', 'sri', 'new', 'the'},
            'suffixes': {'mart', 'store', 'shop', 'traders', 'center', 'centre'},
            'common': {'bhagwan', 'bhagawan', 'krishna', 'ram', 'noor', 'ahmed', 'haji'}
        }

    def find_signboard(self, image):
        print("\n[Signboard] Detecting signboard region...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (3,3), 0)
        
        # More sensitive edge detection
        edges1 = cv2.Canny(blur, 20, 100)  # Lower thresholds
        edges2 = cv2.Canny(blur, 40, 150)
        edges3 = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.bitwise_or(edges, np.uint8(np.absolute(edges3)))
        
        # Connect components
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
        dilated_h = cv2.dilate(edges, kernel_h, iterations=2)
        dilated_v = cv2.dilate(edges, kernel_v, iterations=1)
        dilated = cv2.bitwise_or(dilated_h, dilated_v)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        img_height, img_width = image.shape[:2]
        min_area = max(img_width * img_height * self.min_area_ratio, 
                      self.min_signboard_width * self.min_text_height)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > (img_width * img_height * 0.5):  # Add max area check
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w/h
            
            # More flexible ratio and position constraints
            if 1.0 <= ratio <= 7.0 and y < img_height * 0.6:  # Allow taller signs
                score = self._score_signboard_candidate(enhanced[y:y+h, x:x+w], ratio, area/(img_width*img_height))
                candidates.append((score, (x,y,w,h)))

        if candidates:
            print(f"[Signboard] Found {len(candidates)} candidate regions")
            # Take top 2 candidates and merge if close
            candidates.sort(key=lambda x: x[0], reverse=True)
            if len(candidates) > 1:
                box1 = candidates[0][1]
                box2 = candidates[1][1]
                if self._boxes_are_close(box1, box2):
                    x = min(box1[0], box2[0])
                    y = min(box1[1], box2[1])
                    w = max(box1[0] + box1[2], box2[0] + box2[2]) - x
                    h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
                    best_box = (x, y, w, h)
                else:
                    best_box = candidates[0][1]
            else:
                best_box = candidates[0][1]

            x, y, w, h = best_box
            # Add more padding
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.25)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img_width, x + w + pad_w)
            y2 = min(img_height, y + h + pad_h)
            return image[y1:y2, x1:x2]
        
        # Improved fallback using edge density
        if not candidates:
            # Find region with highest edge density
            edge_sums = np.sum(edges, axis=1) / img_width
            mask = edge_sums > (np.max(edge_sums) * self.edge_density_threshold)
            if np.any(mask):
                y_start = np.argmax(mask)
                y_end = min(y_start + self.max_signboard_height, img_height)
                return image[y_start:y_end, :]
            
            # Final fallback
            return image[0:min(200, img_height), :]

    def _boxes_are_close(self, box1, box2):
        """Check if two bounding boxes are close to each other"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate centers
        c1x = x1 + w1/2
        c1y = y1 + h1/2
        c2x = x2 + w2/2
        c2y = y2 + h2/2
        
        # Check horizontal and vertical distances
        dx = abs(c1x - c2x) / max(w1, w2)
        dy = abs(c1y - c2y) / max(h1, h2)
        
        return dx < 1.5 and dy < 0.8

    def _score_signboard_candidate(self, region, aspect_ratio, area_ratio):
        """Score a potential signboard region"""
        score = 0.0
        
        # Check contrast
        mean_val = np.mean(region)
        std_val = np.std(region)
        if std_val > 40:  # Good contrast
            score += 0.3
        
        # Check aspect ratio (prefer 2:1 to 3:1)
        if 2.0 <= aspect_ratio <= 3.0:
            score += 0.3
        
        # Check area ratio (prefer 10-30% of image)
        if 0.1 <= area_ratio <= 0.3:
            score += 0.2
        
        # Check for text-like features
        grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        if np.mean(np.abs(grad_x)) > np.mean(np.abs(grad_y)):  # Horizontal gradients
            score += 0.2
            
        return score

    def segment_text_regions(self, image):
        """Robust text region segmentation using multiple methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: MSER with correct parameters
        mser = cv2.MSER_create()
        mser.setMinArea(80)
        mser.setMaxArea(8000)
        
        regions, _ = mser.detectRegions(gray)
        boxes = []
        
        # Filter MSER regions
        for p in regions:
            x, y, w, h = cv2.boundingRect(p)
            if h >= self.min_text_height and w > h:
                boxes.append((y, x, y+h, x+w))
        
        # Method 2: Morphological text detection
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create horizontal kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add morphological contours to boxes
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= self.min_text_height and w > h:
                boxes.append((y, x, y+h, x+w))
        
        # Merge and filter regions
        if not boxes:
            # Fallback: simple horizontal division
            height = gray.shape[0]
            strip_height = height // 3
            boxes = [(i*strip_height, 0, (i+1)*strip_height, gray.shape[1]) 
                    for i in range(3)]
        
        return self.merge_overlapping_regions(boxes)

    def merge_overlapping_regions(self, regions):
        """Merge overlapping text regions with improved clustering"""
        if not regions:
            return []
            
        # Convert to numpy array
        boxes = np.array([[y1, x1, y2, x2] for y1, x1, y2, x2 in regions])
        
        # Cluster by vertical position with dynamic eps
        y_centers = (boxes[:, 0] + boxes[:, 2]) / 2
        y_range = np.max(y_centers) - np.min(y_centers)
        eps = max(self.cluster_distance, y_range * 0.1)  # Dynamic clustering distance
        
        clustering = DBSCAN(eps=eps, min_samples=1).fit(y_centers.reshape(-1, 1))
        
        # Merge clusters
        merged = []
        for label in set(clustering.labels_):
            cluster = boxes[clustering.labels_ == label]
            # Compute cluster bounds with padding
            y1 = max(0, np.min(cluster[:, 0]) - 5)
            x1 = max(0, np.min(cluster[:, 1]) - 5)
            y2 = np.max(cluster[:, 2]) + 5
            x2 = np.max(cluster[:, 3]) + 5
            merged.append((y1, x1, y2, x2))
            
        return sorted(merged, key=lambda r: r[0])

    def preprocess_for_ocr(self, image):
        print("[Preprocess] Starting image preprocessing...")
        try:
            if image is None or image.size == 0:
                return []
                
            h, w = image.shape[:2]
            if h < 10 or w < 10:  # Skip tiny images
                return []
                
            processed = []
            
            # Basic preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            
            # Method 1: Basic enhancement
            enhanced = cv2.equalizeHist(gray)
            processed.append(enhanced)
            
            # Method 2: Adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 10)
            processed.append(thresh)
            
            print(f"[Preprocess] Created {len(processed)} image versions")
            return processed
            
        except Exception as e:
            print(f"[Preprocess] Error: {e}")
            return [image] if image is not None else []

    def extract_text(self, image_path):
        print(f"\n[Extract] Processing image: {image_path}")
        try:
            image = cv2.imread(image_path)
            if image is None:
                print("[Extract] Failed to load image")
                return ["Failed to load image"], self.get_default_image(), "shop"
                
            print("[Extract] Finding signboard...")
            signboard = self.find_signboard(image)
            print("[Extract] Segmenting text regions...")
            text_regions = self.segment_text_regions(signboard)
            print(f"[Extract] Found {len(text_regions)} text regions")
            
            all_texts = []
            for idx, (y1, x1, y2, x2) in enumerate(text_regions):
                print(f"\n[Extract] Processing region {idx+1}/{len(text_regions)}")
                region = signboard[int(y1):int(y2), int(x1):int(x2)]
                if region.size == 0:
                    print("[Extract] Empty region, skipping")
                    continue
                
                results, _ = self.process_region(region)
                if results:
                    texts = [r[0] for r in results]
                    print(f"[Extract] Region {idx+1} texts: {texts}")
                    all_texts.extend(texts)

            if not all_texts:
                print("[Extract] No text detected in any region")
                return ["No text detected"], Image.fromarray(cv2.cvtColor(signboard, cv2.COLOR_BGR2RGB)), "shop"

            final_text = self.structure_text(' '.join(all_texts))
            print(f"[Extract] Final structured text: {final_text}")
            business_type = self.get_business_type(final_text)
            print(f"[Extract] Detected business type: {business_type}")
            
            return [final_text], Image.fromarray(cv2.cvtColor(signboard, cv2.COLOR_BGR2RGB)), business_type
            
        except Exception as e:
            print(f"[Extract] Error: {e}")
            return ["Error processing image"], self.get_default_image(), "shop"

    def get_default_image(self):
        """Create a default blank image for error cases"""
        blank = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(blank, "No image available", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return Image.fromarray(blank)

    def segment_signboard(self, image):
        """Simplified signboard segmentation"""
        print("[Segment] Starting advanced segmentation")
        try:
            if image is None or image.size == 0:
                return []
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simple adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 25, 15)
            
            # Find contours
            contours, _ = cv2.findContours(binary, 
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            segments = []
            img_h, img_w = image.shape[:2]
            min_area = img_w * img_h * 0.01
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area < area < self.segment_params['max_area']:
                    x, y, w, h = cv2.boundingRect(cnt)
                    ratio = w / float(h)
                    if (self.segment_params['min_ratio'] < ratio < 
                        self.segment_params['max_ratio']):
                        segments.append((x, y, w, h))
            
            print(f"[Segment] Found {len(segments)} valid segments")
            return segments
            
        except Exception as e:
            print(f"[Segment] Error: {e}")
            return []

    def process_region(self, region):
        """Enhanced region processing with better error handling"""
        print("\n[Process] Processing region")
        try:
            if region is None or region.size == 0:
                return [], region

            results = []
            h, w = region.shape[:2]
            
            # Process whole region first
            processed = self.preprocess_for_ocr(region)
            for img in processed:
                try:
                    detections = self.easyocr_reader.readtext(img, **self.ocr_params)
                    for box, text, conf in detections:
                        if text.strip() and conf > self.easyocr_conf:
                            # Convert box to numpy array properly
                            box_arr = np.array(box, dtype=np.int32)
                            results.append((text.strip(), conf, box_arr))
                except Exception as e:
                    print(f"[Process] Detection error: {e}")
                    continue

            # Cluster results by lines
            if results:
                results = self.cluster_text_by_position(results)

            return results, region

        except Exception as e:
            print(f"[Process] Critical error: {e}")
            return [], region

    def cluster_text_by_position(self, results):
        """Cluster text by vertical position with NumPy 2.0 compatibility"""
        if not results:
            return []

        try:
            # Sort by vertical position
            results = sorted(results, key=lambda x: np.mean(x[2][:, 1]))
            
            # Group by vertical proximity
            lines = []
            current_line = [results[0]]
            mean_height = np.mean([np.max(r[2][:, 1]) - np.min(r[2][:, 1]) for r in results])
            
            for res in results[1:]:
                curr_y = np.mean(current_line[-1][2][:, 1])
                next_y = np.mean(res[2][:, 1])
                
                if abs(next_y - curr_y) <= mean_height * 0.5:
                    current_line.append(res)
                else:
                    # Sort line by horizontal position
                    current_line.sort(key=lambda x: np.min(x[2][:, 0]))
                    lines.extend(current_line)
                    current_line = [res]
            
            # Add last line
            if current_line:
                current_line.sort(key=lambda x: np.min(x[2][:, 0]))
                lines.extend(current_line)

            return lines
            
        except Exception as e:
            print(f"[Cluster] Error: {e}")
            return results

    def enhance_bakery_results(self, results):
        """Enhance bakery-specific detections with NumPy 2.0 compatibility"""
        if not results:
            return results
            
        enhanced = []
        try:
            # Calculate mean height using numpy operations
            heights = [np.max(r[2][:, 1]) - np.min(r[2][:, 1]) for r in results]
            mean_height = np.mean(heights) if heights else 0
            
            for text, conf, box in results:
                # Check for bakery-specific patterns
                for pattern in self.bakery_params['name_patterns']:
                    if re.search(pattern, text):
                        conf *= 1.3
                
                # Get text height
                height = np.max(box[:, 1]) - np.min(box[:, 1])
                
                # Apply size-based boost
                if height > mean_height * 1.2:
                    conf *= self.bakery_params['layout_weights']['large_text']
                
                enhanced.append((text, conf, box))
            
            return sorted(enhanced, key=lambda x: -x[1])
            
        except Exception as e:
            print(f"[Enhance] Error: {e}")
            return results

    def clean_detected_text(self, text):
        print(f"[Clean] Input text: {text}")
        cleaned = re.sub(r'[^a-zA-Z0-9\s&-]', '', text)
        cleaned = ' '.join(cleaned.split())
        print(f"[Clean] Cleaned text: {cleaned}")
        return cleaned

    def filter_duplicate_detections(self, results):
        """Remove duplicate detections keeping highest confidence"""
        if not results:
            return []
            
        # Group by similar text
        text_groups = {}
        for text, conf, box in results:
            text_lower = text.lower()
            if text_lower not in text_groups or conf > text_groups[text_lower][1]:
                text_groups[text_lower] = (text, conf, box)
        
        # Sort by confidence
        return sorted(text_groups.values(), key=lambda x: -x[1])

    def draw_detection(self, image, box, text, conf):
        """Draw detection with improved visibility"""
        cv2.polylines(image, [box], True, self.box_color, 2)
        
        # Add background for better text visibility
        x, y = box[0]
        y = max(30, y - 10)
        cv2.rectangle(image, (x, y-20), (x+len(text)*10, y), (255,255,255), -1)
        
        # Draw text and confidence
        cv2.putText(image, f"{text} ({conf:.2f})", 
                   (x, y-5), self.font, 0.5, (0,0,0), 2)

    def sort_results(self, results):
        """Safely sort OCR results"""
        if not results:
            return []
            
        def get_x(result):
            try:
                box = result[2]
                if box is not None and isinstance(box, np.ndarray):
                    return box[0][0]  # Get leftmost x-coordinate
                return float('inf')  # Put results without valid boxes at the end
            except:
                return float('inf')
        
        return sorted(results, key=get_x)

    def is_valid_text(self, text):
        """Improved text validation"""
        text = text.strip()
        if not text or len(text) < 2:
            return False
            
        # Must contain letters
        if not any(c.isalpha() for c in text):
            return False
            
        # Remove common noise
        text = re.sub(r'[^\w\s]', '', text)
        
        # Check valid character ratio
        valid_chars = sum(1 for c in text if c.isalnum() or c.isspace())
        return valid_chars / len(text) > 0.5

    def clean_text(self, text_lines):
        # Basic text cleaning
        cleaned = []
        seen = set()

        for line in text_lines:
            # Remove noise and normalize
            line = re.sub(r'[^A-Za-z0-9\s&.]', '', line)
            line = ' '.join(line.split())
            
            if line and line.lower() not in seen:
                cleaned.append(line)
                seen.add(line.lower())
        
        # Find most likely shop name
        for line in cleaned:
            words = line.split()
            if len(words) >= 2 and any(w.lower() in self.business_words for w in words):
                return line.title()
        
        return cleaned[0] if cleaned else "Unknown Shop"

    def get_business_type(self, text):
        print(f"[Business Type] Analyzing text: {text}")
        text_lower = text.lower()
        max_score = 0
        best_type = "shop"  # Default fallback
        
        # Score each business category
        for category, base_score in self.business_categories.items():
            if category in text_lower:
                score = base_score
                # Boost score if it appears multiple times or in specific contexts
                if f"{category}s" in text_lower:  # Plural form
                    score += 1
                if any(word in text_lower for word in ["new", "sri", "shree", "the"]):
                    score += 1
                
                if score > max_score:
                    max_score = score
                    best_type = category
        
        print(f"[Business Type] Detected type: {best_type} with score: {max_score}")
        return best_type

    def _process_text(self, texts):
        """Enhanced text processing with BERT correction"""
        if not texts:
            return None

        # First pass: Clean and normalize
        cleaned = []
        for text in texts:
            # Remove numbers, phone numbers and special chars
            text = re.sub(r'\d[-\d\s,]*\d', '', text)  # Remove phone numbers
            text = re.sub(r'[^A-Za-z\s]', ' ', text)   # Keep only letters
            text = ' '.join(text.split())               # Normalize spaces
            
            if text and len(text) >= 3:
                cleaned.append(text.upper())

        # Remove duplicates and sort by length
        cleaned = sorted(set(cleaned), key=len, reverse=True)
        
        # Second pass: BERT correction for each word
        business_name = []
        business_type = []

        for text in cleaned:
            words = text.split()
            for word in words:
                if len(word) < 3:
                    continue
                    
                # Skip if already processed
                if word in business_name or word in business_type:
                    continue

                # Check if it's a known business type
                is_type = False
                for type_words in self.business_keywords.values():
                    if any(fuzz.ratio(word.lower(), kw) > 80 for kw in type_words):
                        business_type.append(word)
                        is_type = True
                        break

                if not is_type:
                    # Use BERT for spelling correction
                    corrected = self._bert_spell_check(word)
                    if corrected and len(corrected) >= 3:
                        business_name.append(corrected)

        # Structure final text
        final_parts = []
        
        # Add business name (up to 2 parts)
        if business_name:
            final_parts.extend(business_name[:2])
        
        # Add business type
        if business_type:
            final_parts.extend(business_type)
            
        final_text = ' '.join(final_parts)
        print(f"[Process] Final text: {final_text}")
        return final_text

    def _remove_noise(self, text):
        """Remove noise from text"""
        # Remove phone numbers and digits
        text = re.sub(r'\d[-\d\s,]*\d', '', text)
        text = re.sub(r'[^A-Za-z\s]', ' ', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

    def _bert_spell_check(self, word):
        """Balanced BERT spell checking"""
        try:
            # Skip short words
            if len(word) < self.min_text_length:
                return word

            # Check business corrections first
            if word.upper() in self.business_corrections:
                return self.business_corrections[word.upper()]

            # Skip if word is a known business term
            if any(word.lower() in keywords for keywords in self.business_keywords.values()):
                return word.upper()

            # Apply BERT correction
            inputs = self.tokenizer(word, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.spell_checker(**inputs)
                predictions = outputs.logits[0]
                
                # Get top prediction
                probs = torch.softmax(predictions, dim=-1)
                values, indices = torch.topk(probs, k=1)
                
                if values[0][0] > self.min_word_confidence:
                    correction = self.tokenizer.decode([indices[0][0]])
                    # Verify similarity
                    if fuzz.ratio(word.lower(), correction.lower()) > 70:
                        return correction.upper()

            return word.upper()
            
        except Exception as e:
            print(f"[Spell] Error: {str(e)}")
            return word.upper()

    def structure_text(self, text):
        """Enhanced text structuring with BERT verification"""
        print(f"[Structure] Input text: {text}")
        try:
            # First pass: Clean and group words
            words = []
            for word in text.split():
                # Remove noise
                word = re.sub(r'[^A-Za-z\s]', '', word)
                word = word.strip().upper()
                
                if len(word) >= 3 and not any(c.isdigit() for c in word):
                    # Group similar words
                    similar_found = False
                    for existing in words:
                        if fuzz.ratio(word, existing) > 85:
                            similar_found = True
                            break
                    if not similar_found:
                        words.append(word)

            # Second pass: Identify business name and type
            name_words = []
            type_words = []
            
            for word in words:
                # Check if it's a shop type word
                is_type = False
                for type_name, keywords in self.shop_keywords.items():
                    if any(fuzz.ratio(word.lower(), kw) > 80 for kw in keywords):
                        corrected = self._get_best_match(word, keywords)
                        if corrected and corrected not in type_words:
                            type_words.append(corrected)
                            is_type = True
                            break
                
                # If not a type, consider it part of business name
                if not is_type:
                    corrected = self._bert_verify_word(word)
                    if corrected and len(corrected) >= 3:
                        name_words.append(corrected)

            # Build final text
            final_words = []
            
            # Add verified name words (up to 2)
            if name_words:
                final_words.extend(name_words[:2])
            
            # Add type words
            final_words.extend(type_words)
            
            final_text = ' '.join(final_words)
            print(f"[Structure] Final text: {final_text}")
            return final_text if final_text else "Unknown"
            
        except Exception as e:
            print(f"[Structure] Error: {str(e)}")
            return "Unknown"

    def _bert_verify_word(self, word):
        """Use BERT to verify and correct business words"""
        try:
            # Check common corrections first
            if word in self.text_corrections:
                return self.text_corrections[word]

            # Skip known valid words
            for keywords in self.shop_keywords.values():
                if word.lower() in keywords:
                    return word

            # Use BERT for verification
            inputs = self.tokenizer(word, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.spell_checker(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1)
                
                # Get top 3 predictions
                values, indices = torch.topk(probs, k=3)
                
                for val, idx in zip(values[0], indices[0]):
                    if val > 0.3:  # Lower confidence threshold
                        correction = self.tokenizer.decode([idx])
                        # Check if correction is reasonable
                        if (len(correction) >= 3 and 
                            fuzz.ratio(word.lower(), correction.lower()) > 60):
                            return correction.upper()
            
            return word

        except Exception as e:
            print(f"[BERT] Error: {str(e)}")
            return word

    def _get_best_match(self, word, keywords):
        """Get best matching word from keywords"""
        best_match = None
        best_ratio = 0
        
        for kw in keywords:
            ratio = fuzz.ratio(word.lower(), kw)
            if ratio > best_ratio and ratio > 80:
                best_ratio = ratio
                best_match = kw.upper()
        
        return best_match if best_match else word