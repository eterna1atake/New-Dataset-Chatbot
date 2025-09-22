import os
import time
import json
import re
import google.generativeai as genai
import streamlit as st
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import ResourceExhausted
import traceback

# Configure API
genai.configure(api_key="AIzaSyBmch5pqGDnzYxmk90cdiK3Z7LuYZry-78")

# Generation config - ปรับให้เสถียรมากขึ้น
generation_config = {
    "temperature": 0.1,  # เพิ่มขึ้นเล็กน้อยเพื่อความเสถียร
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2048,  # ลดลงเพื่อประหยัด token
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# ขยาย Synonym dictionary
SEARCH_KEYWORDS = {
    'ม6_สายศิลป์': ['ม.6 สายศิลป์', 'ม6 สายศิลป์', 'มัธยม 6 ศิลป์', 'ม.๖ ศิลป์', 'สายศิลปศาสตร์', 'ศิลป์คำนวณ'],
    'สายศิลป์': ['ศิลปศาสตร์', 'ศิลป์', 'liberal arts', 'humanities', 'ศิลป์คำนวณ', 'ศิลป์-คำนวณ'],
    'สายวิทย์': ['วิทยาศาสตร์', 'วิทย์', 'science', 'วิทยาศาสตร์-คณิตศาสตร์'],
    'ต่อ': ['เข้าเรียน', 'เข้าศึกษา', 'สมัคร', 'เรียนต่อ', 'ศึกษาต่อ'],
    'หลักสูตร': ['สาขา', 'สาขาวิชา', 'โปรแกรม', 'วิชาเอก', 'แขนง', 'ปริญญา'],
    'เงื่อนไข': ['คุณสมบัติ', 'ข้อกำหนด', 'เกณฑ์', 'เกรด', 'GPAX'],
    'ค่าธรรมเนียม': ['ค่าเทอม', 'ค่าใช้จ่าย', 'เงินเทอม', 'tuition'],
    'คณะ': ['คณะวิศวกรรมศาสตร์', 'คณะวิทยาศาสตร์ประยุกต์', 'คณะเทคโนโลยี', 'คณะศิลปศาสตร์']
}

# System prompt ที่ปรับปรุงแล้ว
SYSTEM_PROMPT = """
คุณเป็นระบบให้ข้อมูลของมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (KMUTNB)

หลักการตอบ:
1. ตอบข้อมูลตรงจากข้อมูลที่ให้มาเท่านั้น
2. ห้ามใช้คำว่า "เอกสาร" "ข้อมูล" "ระบุ" หรือคำที่บอกแหล่งที่มา
3. ตอบแบบทางการ ไม่ใช้ "ครับ" "ค่ะ"
4. ระบุรายละเอียดที่ชัดเจน เช่น ชื่อหลักสูตร คุณสมบัติ ค่าธรรมเนียม
5. จัดรูปแบบให้อ่านง่าย
6. หากไม่มีข้อมูลที่ตรงกับคำถาม ให้ตอบว่า "ไม่มีข้อมูลเรื่องนี้"

ตัวอย่าง:
❌ "จากเอกสารระบุว่า..."
✅ "หลักสูตรที่รับนักเรียน ม.6 สายศิลป์ ได้แก่..."
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=SAFETY_SETTINGS,
    generation_config=generation_config,
    system_instruction=SYSTEM_PROMPT,
)

# Enhanced Rate limiting
class RateLimiter:
    def __init__(self):
        if 'api_calls' not in st.session_state:
            st.session_state.api_calls = []
        if 'last_error_time' not in st.session_state:
            st.session_state.last_error_time = 0
    
    def can_make_request(self):
        current_time = time.time()
        
        # ถ้าเพิ่ง error ให้รอ 30 วินาที
        if current_time - st.session_state.last_error_time < 30:
            return False
            
        # ลบ request เก่าที่เกิน 1 นาที
        st.session_state.api_calls = [
            call_time for call_time in st.session_state.api_calls 
            if current_time - call_time < 60
        ]
        return len(st.session_state.api_calls) < 10  # ลดจำนวนลง
    
    def add_request(self):
        st.session_state.api_calls.append(time.time())
    
    def add_error(self):
        st.session_state.last_error_time = time.time()
    
    def time_until_next_request(self):
        current_time = time.time()
        
        # ถ้าเพิ่ง error
        if current_time - st.session_state.last_error_time < 30:
            return 30 - (current_time - st.session_state.last_error_time)
            
        if not st.session_state.api_calls:
            return 0
        oldest_call = min(st.session_state.api_calls)
        return max(0, 60 - (current_time - oldest_call))

rate_limiter = RateLimiter()

def read_full_document(file_path: str) -> str:
    """อ่านไฟล์ทั้งหมดอย่างปลอดภัย"""
    try:
        if not os.path.exists(file_path):
            return "Error: File not found"
            
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif file_path.lower().endswith('.pdf'):
            try:
                import fitz
                doc = fitz.open(file_path)
                
                full_text = ""
                total_pages = len(doc)
                
                for page_num in range(total_pages):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()
                        
                        if page_text.strip():  # เฉพาะหน้าที่มีเนื้อหา
                            full_text += f"\n--- หน้าที่ {page_num + 1} ---\n"
                            full_text += page_text
                            full_text += f"\n--- จบหน้าที่ {page_num + 1} ---\n"
                            
                    except Exception as e:
                        continue  # ข้ามหน้าที่มีปัญหา
                
                doc.close()
                
                if not full_text.strip():
                    return "Error: ไม่สามารถอ่านเนื้อหาจาก PDF ได้"
                    
                return full_text
                
            except ImportError:
                return "Error: ต้องติดตั้ง PyMuPDF ก่อน (pip install PyMuPDF)"
            except Exception as e:
                return f"Error: ไม่สามารถอ่าน PDF ได้ - {str(e)}"
        else:
            return "Error: รองรับเฉพาะไฟล์ .txt และ .pdf"
            
    except Exception as e:
        return f"Error: {str(e)}"

def extract_relevant_sections(content: str, query: str) -> str:
    """ดึงเนื้อหาที่เกี่ยวข้องจากทั้งไฟล์"""
    if not content or content.startswith("Error:"):
        return content
    
    query_lower = query.lower()
    
    # สร้างรายการคำค้นหา
    search_terms = set()
    words = re.findall(r'\b\w+\b', query_lower)
    
    for word in words:
        search_terms.add(word)
        
        # ค้นหาใน keyword dictionary
        for category, keywords in SEARCH_KEYWORDS.items():
            for keyword in keywords:
                if word in keyword.lower() or keyword.lower() in word:
                    search_terms.update([k.lower() for k in keywords])
                    break
    
    # แยกเนื้อหาเป็นส่วนๆ
    sections = re.split(r'--- หน้าที่ \d+ ---', content)
    relevant_sections = []
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        section_lower = section.lower()
        score = 0
        
        # คำนวณคะแนนความเกี่ยวข้อง
        for term in search_terms:
            if term in section_lower:
                score += section_lower.count(term) * len(term)
        
        # เพิ่มคะแนนสำหรับคำสำคัญ
        important_keywords = ['หลักสูตร', 'สาขา', 'คณะ', 'เงื่อนไข', 'ค่าธรรมเนียม', 'รับสมัคร', 'ม.6', 'สายศิลป์', 'สายวิทย์']
        for keyword in important_keywords:
            if keyword in section_lower:
                score += 100
        
        if score > 0:
            relevant_sections.append((score, section))
    
    # เรียงตามคะแนนและเลือกเนื้อหา
    relevant_sections.sort(reverse=True)
    
    if not relevant_sections:
        # ถ้าไม่เจอ ใช้เนื้อหาจากต้นไฟล์
        return content[:15000]
    
    # รวมเนื้อหาที่เกี่ยวข้อง
    result = ""
    total_length = 0
    max_length = 20000
    
    for score, section in relevant_sections:
        if total_length + len(section) > max_length:
            break
        result += section + "\n\n"
        total_length += len(section)
    
    return result if result else content[:15000]

def clean_response(response: str) -> str:
    """ลบคำที่ไม่ต้องการ"""
    patterns_to_remove = [
        r'จากเอกสาร[^.]*\.?',
        r'ตามเอกสาร[^.]*\.?',
        r'เอกสารระบุ[^.]*\.?',
        r'ตามข้อมูล[^.]*\.?',
        r'ข้อมูลระบุ[^.]*\.?',
        r'ตามที่ระบุ[^.]*\.?',
        r'ขออภัย[^.]*\.?',
        r'\bครับ\b',
        r'\bค่ะ\b',
        r'\bนะ\b',
        r'\bเนอะ\b',
    ]
    
    for pattern in patterns_to_remove:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE)
    
    # ทำความสะอาด
    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
    response = re.sub(r' +', ' ', response)
    return response.strip()

def generate_response(prompt: str, document_content: str) -> str:
    """สร้างคำตอบจาก dataset"""
    if not document_content or document_content.startswith("Error:"):
        return "ไม่สามารถอ่านไฟล์ข้อมูลได้"
    
    # ดึงเนื้อหาที่เกี่ยวข้อง
    relevant_content = extract_relevant_sections(document_content, prompt)
    
    def make_api_call():
        instruction = f"""
คำถาม: {prompt}

ข้อมูล KMUTNB:
{relevant_content[:15000]}

ตอบตรงจากข้อมูลที่ให้มา (ไม่ใช้คำว่า "เอกสาร" หรือ "ครับ/ค่ะ"):"""
        
        try:
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(instruction)
            return clean_response(response.text)
        except Exception as e:
            raise e
    
    try:
        if not rate_limiter.can_make_request():
            wait_time = rate_limiter.time_until_next_request()
            return f"กรุณารอ {wait_time:.0f} วินาที ก่อนถามคำถามใหม่"
        
        result = make_api_call()
        rate_limiter.add_request()
        
        if not result.strip():
            return "ไม่มีข้อมูลเรื่องนี้"
            
        return result
        
    except ResourceExhausted as e:
        rate_limiter.add_error()
        return "API quota เต็ม กรุณารอสักครู่"
        
    except Exception as e:
        rate_limiter.add_error()
        error_msg = str(e).lower()
        
        if "quota" in error_msg or "limit" in error_msg:
            return "API quota เต็ม กรุณารอสักครู่"
        elif "safety" in error_msg:
            return "ไม่สามารถตอบคำถามนี้ได้"
        else:
            return "ไม่สามารถประมวลผลได้ในขณะนี้"

def clear_history():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "สอบถามข้อมูลเกี่ยวกับมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ"}
    ]
    st.rerun()

# Page config
st.set_page_config(
    page_title="KMUTNB Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    
    file_path = st.text_input(
        "เส้นทางไฟล์:", 
        value="FinalDataset.pdf"
    )
    
    if st.button("🔄 โหลดใหม่", use_container_width=True):
        if 'document_content' in st.session_state:
            del st.session_state.document_content
        st.rerun()
    
    if st.button("🗑️ ล้างประวัติ", use_container_width=True):
        clear_history()
    
    # แสดงสถานะ
    if 'document_content' in st.session_state:
        content = st.session_state.document_content
        if content and not content.startswith("Error:"):
            st.success(f"📄 โหลดแล้ว: {len(content):,} ตัวอักษร")
        else:
            st.error("❌ ไม่สามารถโหลดไฟล์ได้")

# Main app
st.title("🎓 KMUTNB Chatbot")

# Load document
if 'document_content' not in st.session_state:
    if file_path.strip():
        with st.spinner("กำลังโหลดไฟล์..."):
            content = read_full_document(file_path)
            st.session_state.document_content = content
            
            if content.startswith("Error:"):
                st.error(f"❌ {content}")
            else:
                st.success(f"✅ โหลดสำเร็จ")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "สอบถามข้อมูลเกี่ยวกับมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ"}
    ]

# Display messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("พิมพ์คำถาม..."):
    if 'document_content' not in st.session_state:
        st.error("❌ กรุณาโหลดไฟล์ก่อน")
        st.stop()
    
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูล..."):
            response = generate_response(prompt, st.session_state.document_content)
            st.write(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})