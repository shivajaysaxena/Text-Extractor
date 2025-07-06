import os
import re
from PIL import Image
from fpdf import FPDF
from text_processor import TextProcessor
from text_processor_llm import TextProcessorLLM
import streamlit as st
import numpy as np
import cv2

def safe_latin1(text):
    if not isinstance(text, str):
        text = str(text)
    return text.encode('latin-1', 'replace').decode('latin-1')

def process_images_and_generate_pdf(image_paths, gemini_api_key, output_pdf_path):
    local_processor = TextProcessor()
    llm_processor = TextProcessorLLM(api_key=gemini_api_key)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for idx, img_path in enumerate(image_paths, 1):
        # --- Analyze the whole board, not just detected regions ---
        try:
            image = cv2.imread(img_path)
            if image is None:
                local_text = "No image detected"
                confs = []
            else:
                # Instead of cropping to signboard, process the whole image
                processed_imgs = local_processor.preprocess_for_ocr(image)
                all_texts = []
                all_confs = []
                for img in processed_imgs:
                    try:
                        detections = local_processor.easyocr_reader.readtext(img, **local_processor.ocr_params)
                        for _, text, conf in detections:
                            if text.strip() and conf > local_processor.easyocr_conf:
                                all_texts.append(text.strip())
                                all_confs.append(conf)
                    except Exception:
                        continue
                # Use BERT correction and structuring
                if all_texts:
                    shop_name = local_processor.structure_text(' '.join(all_texts))
                    local_text = shop_name
                    confs = all_confs
                else:
                    local_text = "No text detected"
                    confs = []
        except Exception:
            local_text = "Error processing image"
            confs = []

        local_conf_max = f"{max(confs):.2f}" if confs else "N/A"
        local_conf_avg = f"{np.mean(confs):.2f}" if confs else "N/A"

        # --- Gemini LLM OCR ---
        llm_texts, _, _ = llm_processor.extract_text_sync(img_path)
        llm_conf = "N/A"

        # --- Add to PDF ---
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, safe_latin1(f"Image {idx}: {os.path.basename(img_path)}"), ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, safe_latin1("Local OCR Extraction:"), ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, safe_latin1(f"Shop Name:\n{local_text}"))
        pdf.cell(0, 10, safe_latin1(f"Max Confidence: {local_conf_max}"), ln=True)
        pdf.cell(0, 10, safe_latin1(f"Avg Confidence: {local_conf_avg}"), ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, safe_latin1("Gemini LLM Extraction:"), ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, safe_latin1(f"Text:\n{llm_texts[0] if llm_texts else 'N/A'}"))
        pdf.cell(0, 10, safe_latin1(f"Confidence: {llm_conf}"), ln=True)
    pdf.output(output_pdf_path)
    print(f"PDF report generated: {output_pdf_path}")

# Example usage:
# image_paths = ["img1.jpg", "img2.jpg", ...]
# process_images_and_generate_pdf(image_paths, gemini_api_key="YOUR_KEY", output_pdf_path="output_report.pdf")
    pdf.ln(5)

    pdf.output(output_pdf_path)
    print(f"PDF report generated: {output_pdf_path}")

# Example usage:
# image_paths = ["img1.jpg", "img2.jpg", ...]
# process_images_and_generate_pdf(image_paths, gemini_api_key="YOUR_KEY", output_pdf_path="output_report.pdf")
