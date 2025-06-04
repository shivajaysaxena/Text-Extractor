# 🏪 Shop Text Detection System

Welcome to the Shop Text Detection System! This application provides intelligent text extraction and analysis from shop signboards using Google's Gemini AI. Built with Python and Streamlit, this tool offers a powerful web interface for seamless shop text detection.

## 🚀 Features

- 📸 **Image Processing**: Real-time text extraction from shop signboards
- 🤖 **AI-Powered Analysis**: Advanced text recognition using Gemini AI
- 🔄 **Text Correction**: Intelligent spelling and format correction
- 💾 **Database Integration**: Store and organize detected shop information
- 🔍 **Search Capability**: Find shops by detected text
- 📊 **Visual Results**: Clean presentation of processed images and text

## 🛠️ Installation

1. **Clone the Repository:**
```bash
git clone <repository-url>
cd project
```

2. **System Requirements:**
- Python 3.9+ (3.10 recommended)
- Version Support:
  - ✅ Python 3.9-3.11: Fully compatible
  - ✅ Python 3.10: Recommended version
  - ❌ Python 3.12+: Not supported by dependencies

3. **Create Virtual Environment:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

4. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

5. **Configure Gemini API:**
- Get API key from Google AI Studio
- Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_api_key_here"
```

## 💻 Usage

1. **Start the Application:**
```bash
streamlit run app.py
```

2. **Key Features:**
- Upload shop images
- Choose between local or LLM-based processing
- View extracted text and shop type
- Search through processed images
- Browse common detected words

## 📁 Project Structure
```
project/
├── app.py                # Main Streamlit application
├── text_processor.py     # Local text processing
├── text_processor_llm.py # Gemini AI integration
├── requirements.txt      # Python dependencies
├── .streamlit/          # Streamlit configuration
│   └── secrets.toml     # API keys
└── README.md            # Documentation
```

## 🧰 Core Dependencies

- **Streamlit**: Web interface framework
- **Google Gemini AI**: Advanced text analysis
- **OpenCV**: Image processing
- **SQLite**: Database management
- **NumPy**: Numerical computations
- **Pillow**: Image handling

## 🤝 Contributing

1. Fork the repository
2. Create feature branch:
```bash
git checkout -b feature/YourFeature
```
3. Commit changes:
```bash
git commit -m "Add: feature description"
```
4. Push to branch:
```bash
git push origin feature/YourFeature
```
5. Open Pull Request

## 📝 License

MIT License - feel free to use and modify for your purposes!

## 🙏 Acknowledgments

- Google Gemini AI for powerful text processing
- Streamlit framework for the interactive interface
- OpenCV community for image processing tools