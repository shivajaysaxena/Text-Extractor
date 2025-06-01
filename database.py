import sqlite3
import os
from datetime import datetime

class ImageDatabase:
    def __init__(self):
        # Don't delete existing database, just connect
        self.conn = sqlite3.connect('images.db', check_same_thread=False)
        self.create_tables()
        # Verify database connection
        count = self.conn.execute('SELECT COUNT(*) FROM images').fetchone()[0]
        print(f"Database initialized with {count} existing records")  # Debug print

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                extracted_text TEXT,
                common_word TEXT NOT NULL,
                image_hash TEXT UNIQUE,
                created_at TIMESTAMP
            )
        ''')
        self.conn.commit()

    def save_image_data(self, image_path, extracted_text, common_word, image_hash):
        cursor = self.conn.cursor()
        try:
            # Handle organized text format
            if isinstance(extracted_text, list):
                extracted_text = '\n'.join(str(t) for t in extracted_text)
            
            # Normalize text before saving
            common_word = str(common_word).lower().strip()
            extracted_text = str(extracted_text).strip()
            
            print(f"Saving - Common word: {common_word}, Hash: {image_hash}")  # Debug
            
            cursor.execute('''
                INSERT INTO images (image_path, extracted_text, common_word, image_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_path, extracted_text, common_word, image_hash, datetime.now()))
            self.conn.commit()
            
            # Verify the save
            cursor.execute('SELECT * FROM images WHERE image_hash = ?', (image_hash,))
            saved_data = cursor.fetchone()
            if saved_data:
                print(f"Saved data: {saved_data}")  # Debug
                return True
            return False
        except Exception as e:
            print(f"Error saving to database: {e}")  # Debug
            return False

    def get_common_words(self):
        cursor = self.conn.cursor()
        try:
            # Improved query to ensure we get results
            cursor.execute('''
                SELECT DISTINCT common_word, COUNT(*) as count
                FROM images 
                WHERE common_word IS NOT NULL 
                AND length(common_word) > 0
                GROUP BY common_word
                ORDER BY count DESC, common_word ASC
            ''')
            words = [row[0].lower() for row in cursor.fetchall() if row[0]]
            print(f"Found {len(words)} common words: {words}")  # Debug
            return words
        except Exception as e:
            print(f"Error getting common words: {e}")
            cursor.execute('SELECT * FROM images')  # Debug - check table content
            print(f"Total records: {len(cursor.fetchall())}")
            return []

    def check_image_exists(self, image_hash):
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM images WHERE image_hash = ?', (image_hash,))
        count = cursor.fetchone()[0]
        print(f"Found {count} matching images for hash: {image_hash}")  # Debug print
        return count > 0

    def get_images_by_common_word(self, common_word):
        cursor = self.conn.cursor()
        try:
            search_term = common_word.lower().strip()
            # Improved search query to check partial matches and extracted text
            cursor.execute('''
                SELECT DISTINCT image_path, extracted_text 
                FROM images 
                WHERE LOWER(common_word) LIKE ? 
                OR LOWER(extracted_text) LIKE ? 
                OR ? LIKE '%' || LOWER(common_word) || '%'
                ORDER BY created_at DESC
            ''', (f"%{search_term}%", f"%{search_term}%", search_term))
            return cursor.fetchall()
        except Exception as e:
            print(f"Error searching images: {e}")
            return []