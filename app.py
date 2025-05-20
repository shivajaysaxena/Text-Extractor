import streamlit as st
import os
from PIL import Image
from database import ImageDatabase
from text_processor import TextProcessor
import shutil
import hashlib

def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def save_uploaded_file(uploaded_file):
    save_dir = "uploaded_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def upload_page(db, processor):
    st.header("Upload and Extract Text")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Extract Text"):
            try:
                temp_path = save_uploaded_file(uploaded_file)
                st.info(f"File saved at: {temp_path}")
                
                image_hash = get_image_hash(temp_path)
                st.info(f"Image hash calculated: {image_hash[:10]}...")
                
                # Check if image already exists
                if db.check_image_exists(image_hash):
                    st.warning("This image has already been processed!")
                    os.remove(temp_path)  # Remove temporary file
                    return
                
                extracted_text = processor.extract_text(temp_path)
                if not extracted_text:
                    st.error("No text could be extracted from the image")
                    return
                    
                st.write("Extracted Text:")
                st.write(extracted_text)
                
                common_word = processor.find_common_word(extracted_text)
                if common_word:
                    st.write(f"Common Word: {common_word}")
                    success = db.save_image_data(temp_path, extracted_text, common_word, image_hash)
                    if success:
                        st.success("Image processed and saved successfully!")
                    else:
                        st.error("Failed to save to database")
                else:
                    st.warning("No common word could be found in the extracted text")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def search_page(db):
    st.header("Search Images by Common Word")
    common_words = db.get_common_words()
    
    # Debug information
    st.write("Debug Information:")
    st.write(f"Number of common words found: {len(common_words)}")
    if not common_words:
        st.info("No words available in database. Please upload and process some images first.")
        
    selected_word = st.selectbox("Select a common word", 
                               ["Select a word..."] + (common_words if common_words else ["No words available"]))
    
    if selected_word and selected_word not in ["Select a word...", "No words available"]:
        images = db.get_images_by_common_word(selected_word)
        if images:
            for img_path, text in images:
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), caption=text, use_column_width=True)
        else:
            st.info("No images found for this word")

def main():
    st.title("Image Text Extractor and Organizer")
    
    db = ImageDatabase()
    processor = TextProcessor()
    
    # Add sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload", "Search"])
    
    if page == "Upload":
        upload_page(db, processor)
    else:
        search_page(db)

if __name__ == "__main__":
    main()
