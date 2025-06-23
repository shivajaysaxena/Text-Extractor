import streamlit as st
import os
from PIL import Image
from database import ImageDatabase
from text_processor import TextProcessor
from text_processor_llm import TextProcessorLLM
import shutil
import hashlib
import asyncio

# Initialize processors
@st.cache_resource
def get_processors():
    local_processor = TextProcessor()
    llm_processor = TextProcessorLLM(api_key=st.secrets["GEMINI_API_KEY"])
    return local_processor, llm_processor

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

async def process_with_llm(processor, temp_path):
    """Helper function to process with LLM"""
    return await processor.extract_text(temp_path)

def upload_page(db, processor):
    st.header("Upload and Extract Text")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Extract Text"):
            temp_path = save_uploaded_file(uploaded_file)
            
            # Handle both sync and async processors
            if isinstance(processor, TextProcessorLLM):
                # Use sync wrapper
                shops_text, visualized, common_phrase = processor.extract_text_sync(temp_path)
            else:
                # Use sync processor
                shops_text, visualized, common_phrase = processor.extract_text(temp_path)
            
            # Display extracted text for each shop
            st.subheader("Extracted Text:")
            for i, shop_text in enumerate(shops_text, 1):
                with st.expander(f"Shop {i}"):
                    st.write(shop_text)
                    if common_phrase:
                        st.write(f"Key Phrase: {common_phrase}")
                        # Save to database using extracted phrase
                        image_hash = get_image_hash(temp_path)
                        if not db.check_image_exists(image_hash):
                            db.save_image_data(temp_path, shop_text, common_phrase, image_hash)
            
            # Show OCR visualization
            # st.subheader("Text Detection Visualization")
            # st.image(visualized, caption="Detected Text", use_column_width=True)
            
            # # Add download button for OCR visualization
            # with open('temp_ocr_vis.jpg', 'rb') as file:
            #     btn = st.download_button(
            #         label="Download OCR Visualization",
            #         data=file,
            #         file_name="ocr_visualization.jpg",
            #         mime="image/jpeg"
            #     )

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
    local_processor, llm_processor = get_processors()
    
    # Model selection
    model_type = st.radio(
        "Select Processing Model",
        ["Local Model", "LLM Model (Gemini)"],
        help="Choose between local processing or cloud LLM"
    )
    
    # Set processor based on selection
    processor = local_processor if model_type == "Local Model" else llm_processor
    
    # Add sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload", "Search"])
    
    if page == "Upload":
        upload_page(db, processor)
    else:
        search_page(db)

if __name__ == "__main__":
    main()