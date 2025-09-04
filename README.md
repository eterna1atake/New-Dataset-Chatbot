# KMUTNB Chatbot

A Streamlit-based chatbot that provides information about King Mongkut's University of Technology North Bangkok (KMUTNB) based on the Dataset_kmutnb.docx file.

## Features

- 📚 Reads and processes Word documents (.docx files)
- 🤖 AI-powered responses using Google's Gemini model
- 💬 Interactive chat interface with Streamlit
- 📊 Extracts both text and table data from documents
- 🎯 KMUTNB-specific information and responses

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Setup**
   - You'll need to add your Google Generative AI API key to the `app.py` file
   - Replace the empty string in `genai.configure(api_key="")` with your actual API key

3. **Document Setup**
   - Ensure `Dataset_kmutnb.docx` is in the same directory as `app.py`
   - The chatbot will automatically read this file when started

## Usage

1. **Run the Chatbot**
   ```bash
   streamlit run app.py
   ```

2. **Test the Setup**
   ```bash
   python test_chatbot.py
   ```

3. **Access the Interface**
   - Open your browser to the URL shown in the terminal (usually http://localhost:8501)
   - Start chatting with the KMUTNB chatbot!

## Document Content

The chatbot can answer questions about:
- 📋 Admission criteria and requirements
- 🏛️ Faculty and department information
- 📚 Academic programs and courses
- 📅 Application deadlines and procedures
- 🎓 Student qualifications and requirements
- 📝 Required documents and forms

## File Structure

```
workaw/
├── app.py                 # Main Streamlit application
├── document_reader.py     # Word document processing module
├── prompt.py             # AI prompt configuration
├── requirements.txt      # Python dependencies
├── test_chatbot.py       # Testing script
├── README.md            # This file
└── Dataset_kmutnb.docx  # KMUTNB dataset (Word document)
```

## Troubleshooting

- **Document not found**: Ensure `Dataset_kmutnb.docx` is in the correct directory
- **API errors**: Check your Google Generative AI API key
- **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`

## Example Questions

You can ask the chatbot questions like:
- "KMUTNB มีคณะอะไรบ้าง"
- "เกณฑ์การรับสมัครเป็นอย่างไร"
- "ต้องใช้เอกสารอะไรบ้างในการสมัคร"
- "วันสอบคือวันไหน"
- "คุณสมบัติของผู้สมัครเป็นอย่างไร" 