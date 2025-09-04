import os
import time
import json
import re
import google.generativeai as genai
import streamlit as st
from prompt import PROMPT_WORKAW
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import ResourceExhausted
from typing import Dict, List, Any

# Import enhanced document reader
try:
    from document_reader import EnhancedDocumentReader, get_kmutnb_summary, search_in_document
    ENHANCED_READER_AVAILABLE = True
except ImportError:
    ENHANCED_READER_AVAILABLE = False
    # ฟังก์ชันสำรองแบบเดิม
    def get_kmutnb_summary(file_path: str, use_ocr: bool = False, expert_role: str = "") -> str:
        try:
            if file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path.lower().endswith('.pdf'):
                try:
                    import fitz
                    doc = fitz.open(file_path)
                    content = ""
                    for page in doc:
                        content += page.get_text()
                    doc.close()
                except ImportError:
                    return "Error: PyMuPDF not installed. Please install: pip install PyMuPDF"
            else:
                return "Error: Unsupported file type. Please use .txt or .pdf"
            
            if len(content) > 15000:
                content = content[:15000] + "\n\n[เนื้อหาถูกตัดเพื่อประหยัด token]"
            
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

# Configure API
genai.configure(api_key="AIzaSyA4YjD2FBii2N7HtWq4LWtIPpZLthonp6c")

# Enhanced generation config
generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# Enhanced prompt for direct responses
ENHANCED_PROMPT_WORKAW = """
คุณเป็น AI Assistant ของมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (KMUTNB) 

กฎการตอบคำถาม:
1. หากพบข้อมูลในเอกสาร: ให้ตอบข้อมูลโดยตรง ไม่ต้องขอโทษหรือขออภัย
2. หากไม่พบข้อมูลในเอกสาร: ให้ตอบเพียงว่า "ไม่พบข้อมูล"
3. ตอบเป็นภาษาไทยเท่านั้น
4. ใช้ข้อมูลจากเอกสารที่ให้มาเป็นหลัก
5. ตอบแบบกระชับ ชัดเจน ตรงประเด็น
6. ไม่ต้องบอกแหล่งที่มาของข้อมูล เว้นแต่จำเป็น

ตัวอย่างการตอบ:
- เมื่อพบข้อมูล: "หลักสูตรวิศวกรรมศาสตร์มีทั้งหมด 15 สาขา ได้แก่ วิศวกรรมเครื่องกล วิศวกรรมไฟฟ้า..."
- เมื่อไม่พบข้อมูล: "ไม่พบข้อมูล"
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=SAFETY_SETTINGS,
    generation_config=generation_config,
    system_instruction=ENHANCED_PROMPT_WORKAW,
)

# Enhanced Rate limiting with better tracking
class EnhancedRateLimiter:
    def __init__(self):
        if 'api_calls' not in st.session_state:
            st.session_state.api_calls = []
        if 'api_errors' not in st.session_state:
            st.session_state.api_errors = []
        if 'total_tokens_used' not in st.session_state:
            st.session_state.total_tokens_used = 0
    
    def can_make_request(self):
        current_time = time.time()
        st.session_state.api_calls = [
            call_time for call_time in st.session_state.api_calls 
            if current_time - call_time < 60
        ]
        return len(st.session_state.api_calls) < 10
    
    def add_request(self, tokens_used: int = 0):
        st.session_state.api_calls.append(time.time())
        st.session_state.total_tokens_used += tokens_used
    
    def add_error(self, error_msg: str):
        st.session_state.api_errors.append({
            'time': time.time(),
            'error': error_msg
        })
        if len(st.session_state.api_errors) > 10:
            st.session_state.api_errors = st.session_state.api_errors[-10:]
    
    def time_until_next_request(self):
        if not st.session_state.api_calls:
            return 0
        oldest_call = min(st.session_state.api_calls)
        return max(0, 60 - (time.time() - oldest_call))

    def get_recent_errors(self):
        current_time = time.time()
        return [
            error for error in st.session_state.api_errors
            if current_time - error['time'] < 300
        ]

rate_limiter = EnhancedRateLimiter()

# Document management with OCR and expert role support
class DocumentManager:
    def __init__(self):
        if 'document_content' not in st.session_state:
            st.session_state.document_content = None
        if 'document_metadata' not in st.session_state:
            st.session_state.document_metadata = {}
        if 'document_sections' not in st.session_state:
            st.session_state.document_sections = {}
        if 'document_keywords' not in st.session_state:
            st.session_state.document_keywords = []
        if 'last_file_path' not in st.session_state:
            st.session_state.last_file_path = None
        if 'use_ocr' not in st.session_state:
            st.session_state.use_ocr = False
        if 'expert_role' not in st.session_state:
            st.session_state.expert_role = ""
    
    def load_document(self, file_path: str, use_ocr: bool = False, expert_role: str = "") -> tuple[str, str]:
        # ตรวจสอบว่าต้องโหลดใหม่หรือไม่
        needs_reload = (
            st.session_state.last_file_path != file_path or
            st.session_state.use_ocr != use_ocr or
            st.session_state.expert_role != expert_role or
            not st.session_state.document_content
        )
        
        if not needs_reload:
            return st.session_state.document_content, "✅ ใช้เอกสารที่โหลดไว้แล้ว"
        
        if not os.path.exists(file_path):
            search_paths = self._get_search_paths(file_path)
            
            for path in search_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            else:
                return None, f"ไม่พบไฟล์เอกสาร: {file_path}"
        
        try:
            if ENHANCED_READER_AVAILABLE:
                reader = EnhancedDocumentReader(file_path, use_ocr=use_ocr, expert_role=expert_role)
                content = reader.get_comprehensive_summary()
                
                st.session_state.document_metadata = reader.metadata
                st.session_state.document_sections = reader.sections
                st.session_state.document_keywords = list(reader.keywords)
                
                status = f"โหลดเอกสารสำเร็จ (Enhanced Mode)"
                if use_ocr:
                    status += " + OCR"
                if expert_role:
                    status += f" | ผู้เชี่ยวชาญ: {expert_role}"
                status += f" - {len(content):,} ตัวอักษร"
                
                if reader.metadata:
                    status += f" | หน้า: {reader.metadata.get('pages', 'ไม่ทราบ')}"
            else:
                content = get_kmutnb_summary(file_path, use_ocr=use_ocr, expert_role=expert_role)
                status = f"โหลดเอกสารสำเร็จ (Basic Mode) - {len(content):,} ตัวอักษร"
            
            if content.startswith("Error:"):
                return None, content
            
            st.session_state.document_content = content
            st.session_state.last_file_path = file_path
            st.session_state.use_ocr = use_ocr
            st.session_state.expert_role = expert_role
            
            return content, status
            
        except Exception as e:
            error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
            return None, error_msg
    
    def _get_search_paths(self, original_path: str) -> List[str]:
        current_dir = os.path.dirname(__file__) if __file__ else os.getcwd()
        filename = os.path.basename(original_path)
        
        search_paths = [
            original_path,
            os.path.join(current_dir, filename),
            os.path.join(current_dir, "dataset_reseach.pdf"),
            os.path.join(current_dir, "dataset.pdf"),
            os.path.join(current_dir, "data.pdf"),
            os.path.join(current_dir, "kmutnb.pdf"),
            os.path.join(current_dir, "kmutnb_data.pdf"),
            os.path.join(current_dir, "documents", filename),
            os.path.join(current_dir, "data", filename),
        ]
        
        return search_paths
    
    def search_document(self, search_term: str) -> str:
        if not st.session_state.document_content:
            return "ไม่พบข้อมูล"
        
        if ENHANCED_READER_AVAILABLE and st.session_state.last_file_path:
            result = search_in_document(
                st.session_state.last_file_path, 
                search_term, 
                use_ocr=st.session_state.use_ocr
            )
            # ตรวจสอบว่าพบข้อมูลหรือไม่
            if "ไม่พบ" in result or "❌" in result:
                return "ไม่พบข้อมูล"
            return result
        else:
            content = st.session_state.document_content
            lines = content.split('\n')
            found_lines = []
            
            for i, line in enumerate(lines):
                if search_term.lower() in line.lower():
                    context_start = max(0, i-1)
                    context_end = min(len(lines), i+2)
                    context = lines[context_start:context_end]
                    found_lines.append(f"=== บรรทัดที่ {i+1} ===\n" + '\n'.join(context) + "\n")
            
            if found_lines:
                return f"พบคำว่า '{search_term}' ในเอกสาร {len(found_lines)} ตำแหน่ง:\n\n" + '\n'.join(found_lines[:5])
            else:
                return "ไม่พบข้อมูล"

doc_manager = DocumentManager()

def clear_history():
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดีค่ะ สอบถามข้อมูลเกี่ยวกับ KMUTNB เรื่องใดคะ"}
    ]
    st.rerun()

def safe_api_call(api_function, max_retries=3):
    for attempt in range(max_retries):
        try:
            if not rate_limiter.can_make_request():
                wait_time = rate_limiter.time_until_next_request()
                if wait_time > 0:
                    st.warning(f"⏳ รอ {wait_time:.0f} วินาที เนื่องจาก rate limit")
                    time.sleep(wait_time + 1)
            
            result = api_function()
            rate_limiter.add_request(tokens_used=100)
            return result
            
        except ResourceExhausted as e:
            error_msg = f"API quota เกิน (ครั้งที่ {attempt + 1})"
            rate_limiter.add_error(error_msg)
            
            wait_time = 60 * (attempt + 1)
            if attempt < max_retries - 1:
                st.warning(f"⚠️ {error_msg}! รอ {wait_time} วินาที...")
                time.sleep(wait_time)
            else:
                return "ไม่พบข้อมูล"
                
        except Exception as e:
            error_msg = f"API Error: {str(e)}"
            rate_limiter.add_error(error_msg)
            
            if attempt < max_retries - 1:
                st.warning(f"⚠️ {error_msg} (ลองใหม่ครั้งที่ {attempt + 2})")
                time.sleep(5 * (attempt + 1))
            else:
                return "ไม่พบข้อมูล"
    
    return "ไม่พบข้อมูล"

def enhanced_response_generation(prompt: str, document_content: str, expert_role: str = "") -> str:
    question_type = analyze_question_type(prompt)
    enhanced_prompt = enhance_prompt_based_on_type(prompt, question_type, expert_role)
    
    def generate_response():
        history = []
        
        # เพิ่มคำแนะนำเฉพาะการตอบคำถาม
        instruction = """
คำสั่งสำคัญ:
- หากพบข้อมูลในเอกสาร: ตอบข้อมูลโดยตรง ไม่ต้องขอโทษหรือขออภัย
- หากไม่พบข้อมูล: ตอบเพียง "ไม่พบข้อมูล"
- ห้ามใช้คำว่า "ขออภัย" "ขอโทษ" "อย่างไรก็ตาม" "น่าเสียดาย"
- ตอบเป็นภาษาไทยเท่านั้น
        """
        
        history.append({
            "role": "user", 
            "parts": [{"text": f"{instruction}\n\nเอกสารอ้างอิง:\n{document_content}"}]
        })
        
        recent_messages = st.session_state["messages"][-10:]
        for msg in recent_messages:
            history.append({
                "role": msg["role"], 
                "parts": [{"text": msg["content"]}]
            })
        
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(enhanced_prompt)
        
        # ตรวจสอบและทำความสะอาดคำตอบ
        cleaned_response = clean_response(response.text)
        return cleaned_response
    
    return safe_api_call(generate_response)

def clean_response(response_text: str) -> str:
    """ทำความสะอาดคำตอบ เอาคำขออภัยออก"""
    
    # คำที่ไม่ต้องการ
    unwanted_phrases = [
        "ขออภัย", "ขอโทษ", "เสียใจด้วย", "น่าเสียดาย",
        "อย่างไরก็ตาม", "อย่างไรก็ดี", "แต่ทั้งนี้",
        "ขออภัยครับ", "ขออภัยค่ะ", "ขอโทษครับ", "ขอโทษค่ะ"
    ]
    
    # ลบประโยคที่มีคำขออภัย
    sentences = response_text.split('.')
    cleaned_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and not any(phrase in sentence for phrase in unwanted_phrases):
            cleaned_sentences.append(sentence)
    
    cleaned_response = '. '.join(cleaned_sentences)
    
    # ถ้าคำตอบว่างเปล่าหรือไม่มีข้อมูลที่มีประโยชน์
    if not cleaned_response.strip() or len(cleaned_response.strip()) < 10:
        return "ไม่พบข้อมูล"
    
    return cleaned_response.strip()

def analyze_question_type(prompt: str) -> str:
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['ค้นหา', 'หา', 'search', 'find']):
        return 'search'
    elif any(word in prompt_lower for word in ['เปรียบเทียบ', 'compare', 'ต่าง', 'เหมือน']):
        return 'compare'
    elif any(word in prompt_lower for word in ['อธิบาย', 'explain', 'คืออะไร', 'ทำไม', 'อย่างไร']):
        return 'explain'
    elif any(word in prompt_lower for word in ['รายชื่อ', 'list', 'มีอะไรบ้าง', 'ทั้งหมด']):
        return 'list'
    elif any(word in prompt_lower for word in ['ตัวอย่าง', 'example', 'เช่น']):
        return 'example'
    else:
        return 'general'

def enhance_prompt_based_on_type(prompt: str, question_type: str, expert_role: str = "") -> str:
    base_enhancement = ""
    if expert_role:
        base_enhancement = f"คุณเป็นผู้เชี่ยวชาญด้าน {expert_role} "
    
    enhancements = {
        'search': f"{base_enhancement}ค้นหาข้อมูลที่เกี่ยวข้องในเอกสารและตอบอย่างละเอียด: ",
        'compare': f"{base_enhancement}เปรียบเทียบและวิเคราะห์ความแตกต่างอย่างชัดเจน: ",
        'explain': f"{base_enhancement}อธิบายอย่างละเอียดและให้ตัวอย่างประกอบ: ",
        'list': f"{base_enhancement}จัดทำรายการที่ครบถ้วนและเรียงลำดับ: ",
        'example': f"{base_enhancement}ให้ตัวอย่างที่ชัดเจนและหลากหลาย: ",
        'general': f"{base_enhancement}ตอบคำถามอย่างละเอียดและครบถ้วน: "
    }
    
    return enhancements.get(question_type, base_enhancement + "ตอบคำถามอย่างละเอียดและครบถ้วน: ") + prompt

# Page config
st.set_page_config(
    page_title="KMUTNB Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with enhanced controls
with st.sidebar:
    st.header("Settings")
    
    # OCR Settings
    use_ocr = st.checkbox("OCR PDF", value=False, 
                         help="เปิดใช้ OCR เพื่ออ่าน PDF ที่เป็นภาพหรือ scanned document")
    
    # Expert Role Settings
    expert_role = st.text_input("Role", 
                               placeholder="",
                               help="กำหนดบทบาทเพื่อให้การตอบคำถามแม่นยำขึ้น")
    
    # Document Settings
    st.subheader("Document")
    file_path = st.text_input("Path File Document", 
                             value="/Users/zayxaxto/Documents/kmutnb_chatbot/workaw/Last Dataset.pdf",
                             help="ระบุ path ของไฟล์เอกสาร")
    
    if st.button("Reload", use_container_width=True):
        st.session_state.document_content = None
        st.rerun()
    
    if st.button("Clear History", use_container_width=True):
        clear_history()

# Main app
st.title("💬 KMUTNB Enhanced Chatbot")
st.write("ระบบ AI ตอบคำถามเกี่ยวกับ KMUTNB อย่างตรงไปตรงมา")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "model",
            "content": "สวัสดีค่ะ สอบถามข้อมูลเกี่ยวกับ KMUTNB เรื่องใดคะ",
        }
    ]

# Load document with new settings
file_content, load_status = doc_manager.load_document(file_path, use_ocr=use_ocr, expert_role=expert_role)

# Display load status
if file_content is None:
    st.error(f"❌ {load_status}")
    st.info("💡 วิธีแก้ไข: ตรวจสอบ path ของไฟล์หรือวางไฟล์ในโฟลเดอร์เดียวกันกับ app.py")
    
    with st.expander("📁 ตำแหน่งที่ระบบจะค้นหาไฟล์"):
        search_paths = doc_manager._get_search_paths(file_path)
        for path in search_paths:
            status = "✅" if os.path.exists(path) else "❌"
            st.markdown(f"**{status}** `{path}`")
else:
    st.success(f"✅ {load_status}")

# Display messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("💭 Type your question"):
    if file_content is None:
        st.error("❌ กรุณาโหลดเอกสารก่อนใช้งาน")
        st.stop()
    
    if not rate_limiter.can_make_request():
        wait_time = rate_limiter.time_until_next_request()
        st.error(f"⏳ กรุณารอ {wait_time:.0f} วินาที ก่อนส่งข้อความใหม่")
        st.stop()
    
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("model"):
        with st.spinner("🤔 กำลังประมวลผลคำถาม..."):
            
            if prompt.lower().startswith("ค้นหา:") or prompt.lower().startswith("search:"):
                search_term = prompt.split(":", 1)[1].strip()
                response_text = doc_manager.search_document(search_term)
            else:
                response_text = enhanced_response_generation(prompt, file_content, expert_role)
            
            st.write(response_text)
            st.session_state["messages"].append({"role": "model", "content": response_text})
