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
            cursor.execute('''
                INSERT INTO images (image_path, extracted_text, common_word, image_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_path, extracted_text, common_word.lower(), image_hash, datetime.now()))
            self.conn.commit()
            
            # Verify the save operation
            cursor.execute('SELECT COUNT(*) FROM images WHERE image_hash = ?', (image_hash,))
            if cursor.fetchone()[0] > 0:
                print(f"Successfully saved image with hash {image_hash}")  # Debug print
                return True
            return False
        except Exception as e:
            print(f"Error saving to database: {e}")  # Debug print
            return False

    def get_common_words(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute('SELECT DISTINCT common_word FROM images WHERE common_word IS NOT NULL')
            words = [row[0] for row in cursor.fetchall()]
            print(f"Found common words: {words}")  # Debug print
            return words
        except Exception as e:
            print(f"Error getting common words: {e}")  # Debug print
            return []

    def check_image_exists(self, image_hash):
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM images WHERE image_hash = ?', (image_hash,))
        count = cursor.fetchone()[0]
        print(f"Found {count} matching images for hash: {image_hash}")  # Debug print
        return count > 0

    def get_images_by_common_word(self, common_word):
        cursor = self.conn.cursor()
        cursor.execute('SELECT image_path, extracted_text FROM images WHERE common_word = ?', (common_word,))
        return cursor.fetchall()
