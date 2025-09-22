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

# Generation config - ปรับให้ประมวลผลได้มากขึ้น
generation_config = {
    "temperature": 0.1,  # ลดลงเพื่อความแม่นยำ
    "top_p": 0.8,
    "top_k": 20,
    "max_output_tokens": 4096,
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# System prompt ที่เน้นการอ่านทุกอย่างโดยไม่พลาด
SYSTEM_PROMPT = """
คุณเป็นระบบให้ข้อมูลของมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (KMUTNB)

คำสั่งสำคัญที่สุด:
1. อ่านข้อมูลทั้งหมดที่ได้รับอย่างละเอียดครบถ้วน ทุกบรรทัด ทุกคำ
2. ค้นหาข้อมูลที่เกี่ยวข้องจากทุกส่วนของเนื้อหา ไม่พลาดแม้แต่บรรทัดเดียว
3. ตอบข้อมูลตรงจากที่มีในข้อมูลเท่านั้น
4. ห้ามใช้คำว่า "เอกสาร" "ข้อมูล" "ระบุ" "ครับ" "ค่ะ"
5. ระบุรายละเอียดให้ครบถ้วน
6. หากไม่เจอข้อมูลที่ตรงกับคำถาม ให้ตอบ "ไม่มีข้อมูลเรื่องนี้"

สำคัญมาก: ต้องอ่านทุกส่วนของข้อมูลที่ได้รับ อย่าข้าม อย่ามองข้าม
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=SAFETY_SETTINGS,
    generation_config=generation_config,
    system_instruction=SYSTEM_PROMPT,
)

# Rate limiting
class RateLimiter:
    def __init__(self):
        if 'api_calls' not in st.session_state:
            st.session_state.api_calls = []
        if 'last_error_time' not in st.session_state:
            st.session_state.last_error_time = 0
    
    def can_make_request(self):
        current_time = time.time()
        
        if current_time - st.session_state.last_error_time < 30:
            return False
            
        st.session_state.api_calls = [
            call_time for call_time in st.session_state.api_calls 
            if current_time - call_time < 60
        ]
        return len(st.session_state.api_calls) < 5  # ลดลงเพราะส่งข้อมูลเยอะ
    
    def add_request(self):
        st.session_state.api_calls.append(time.time())
    
    def add_error(self):
        st.session_state.last_error_time = time.time()
    
    def time_until_next_request(self):
        current_time = time.time()
        
        if current_time - st.session_state.last_error_time < 30:
            return 30 - (current_time - st.session_state.last_error_time)
            
        if not st.session_state.api_calls:
            return 0
        oldest_call = min(st.session_state.api_calls)
        return max(0, 60 - (current_time - oldest_call))

rate_limiter = RateLimiter()

def read_complete_document(file_path: str) -> str:
    """อ่านไฟล์ทั้งหมดแบบสมบูรณ์ ไม่ตัดทิ้งอะไร"""
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
                
                complete_text = ""
                total_pages = len(doc)
                
                print(f"เริ่มอ่าน PDF ทั้งหมด {total_pages} หน้า")
                
                for page_num in range(total_pages):
                    try:
                        page = doc[page_num]
                        
                        # อ่านทุกอย่างจากหน้านี้
                        page_text = page.get_text("text")
                        
                        if page_text.strip():
                            complete_text += f"\n--- หน้า {page_num + 1} ---\n"
                            complete_text += page_text.strip()
                            complete_text += f"\n--- จบหน้า {page_num + 1} ---\n\n"
                            
                        print(f"อ่านหน้า {page_num + 1}/{total_pages}")
                            
                    except Exception as e:
                        print(f"ข้ามหน้า {page_num + 1}: {str(e)}")
                        continue
                
                doc.close()
                print(f"อ่านครบทั้งหมด {len(complete_text)} ตัวอักษร")
                return complete_text
                
            except ImportError:
                return "Error: ต้องติดตั้ง PyMuPDF ก่อน (pip install PyMuPDF)"
            except Exception as e:
                return f"Error: ไม่สามารถอ่าน PDF ได้ - {str(e)}"
        else:
            return "Error: รองรับเฉพาะไฟล์ .txt และ .pdf"
            
    except Exception as e:
        return f"Error: {str(e)}"

def send_complete_data_to_ai(prompt: str, full_content: str) -> str:
    """ส่งข้อมูลทั้งหมดไปให้ AI โดยแบ่งเป็นชิ้นๆ ถ้าจำเป็น"""
    
    if not full_content or full_content.startswith("Error:"):
        return "ไม่สามารถอ่านไฟล์ข้อมูลได้"
    
    # ตรวจสอบขนาดข้อมูล
    max_size = 80000  # เพิ่มขนาดสูงสุด
    
    if len(full_content) <= max_size:
        # ส่งทั้งหมดเลยถ้าไม่ใหญ่เกินไป
        return make_single_api_call(prompt, full_content)
    else:
        # แบ่งข้อมูลเป็นส่วนๆ และส่งหลายรอบ
        return make_multiple_api_calls(prompt, full_content, max_size)

def make_single_api_call(prompt: str, content: str) -> str:
    """ส่งข้อมูลทั้งหมดในครั้งเดียว"""
    
    instruction = f"""
คำถาม: {prompt}

ข้อมูลทั้งหมดจาก KMUTNB:
{content}

คำสั่ง:
- อ่านข้อมูลทั้งหมดอย่างละเอียด
- ค้นหาข้อมูลที่เกี่ยวข้องกับคำถาม
- ตอบตรงจากข้อมูลที่มี
- ไม่ใช้คำว่า "เอกสาร" "ครับ" "ค่ะ"
"""
    
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(instruction)
        return clean_response(response.text)
    except Exception as e:
        raise e

def make_multiple_api_calls(prompt: str, full_content: str, chunk_size: int) -> str:
    """แบ่งข้อมูลเป็นชิ้นๆ และส่งหลายครั้ง"""
    
    # แบ่งตามหน้า
    pages = re.split(r'--- หน้า \d+ ---', full_content)
    pages = [page.strip() for page in pages if page.strip()]
    
    all_responses = []
    current_chunk = ""
    
    for page in pages:
        if len(current_chunk + page) > chunk_size:
            if current_chunk:
                # ส่งชิ้นปัจจุบัน
                try:
                    response = make_single_api_call(prompt, current_chunk)
                    if response and "ไม่มีข้อมูลเรื่องนี้" not in response:
                        all_responses.append(response)
                except:
                    pass
                current_chunk = ""
        
        current_chunk += page + "\n\n"
    
    # ส่งชิ้นสุดท้าย
    if current_chunk:
        try:
            response = make_single_api_call(prompt, current_chunk)
            if response and "ไม่มีข้อมูลเรื่องนี้" not in response:
                all_responses.append(response)
        except:
            pass
    
    # รวมคำตอบ
    if all_responses:
        return "\n\n".join(all_responses)
    else:
        return "ไม่มีข้อมูลเรื่องนี้"

def clean_response(response: str) -> str:
    """ลบคำที่ไม่ต้องการ"""
    if not response:
        return "ไม่มีข้อมูลเรื่องนี้"
        
    patterns_to_remove = [
        r'จากเอกสาร[^.]*\.?',
        r'ตามเอกสาร[^.]*\.?',
        r'เอกสารระบุ[^.]*\.?',
        r'ตามข้อมูล[^.]*\.?',
        r'ข้อมูลระบุ[^.]*\.?',
        r'ตามที่ระบุ[^.]*\.?',
        r'\bครับ\b',
        r'\bค่ะ\b',
        r'\bนะ\b',
    ]
    
    for pattern in patterns_to_remove:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE)
    
    # ทำความสะอาด
    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
    response = re.sub(r' +', ' ', response)
    return response.strip()

def generate_complete_response(prompt: str, document_content: str) -> str:
    """สร้างคำตอบจากข้อมูลทั้งหมด"""
    
    try:
        if not rate_limiter.can_make_request():
            wait_time = rate_limiter.time_until_next_request()
            return f"กรุณารอ {wait_time:.0f} วินาที ก่อนถามคำถามใหม่"
        
        result = send_complete_data_to_ai(prompt, document_content)
        rate_limiter.add_request()
        
        if not result or not result.strip():
            return "ไม่มีข้อมูลเรื่องนี้"
            
        return result
        
    except ResourceExhausted:
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
            return f"เกิดข้อผิดพลาด: {str(e)}"

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
            pages_count = len(re.findall(r'--- หน้า \d+ ---', content))
            st.success(f"📄 โหลดสำเร็จ!")
            st.info(f"📊 {pages_count} หน้า")
            st.info(f"📝 {len(content):,} ตัวอักษร")
            
            

# Main app
st.title("🎓 KMUTNB Chatbot")
st.caption("อ่านข้อมูลทั้งหมดและส่งให้ AI ประมวลผลทั้งไฟล์")

# Load document
if 'document_content' not in st.session_state:
    if file_path.strip():
        with st.spinner("กำลังอ่านไฟล์ทั้งหมด..."):
            content = read_complete_document(file_path)
            st.session_state.document_content = content
            
            if content.startswith("Error:"):
                st.error(f"❌ {content}")
            else:
                pages_count = len(re.findall(r'--- หน้า \d+ ---', content))
                st.success(f"✅ อ่านไฟล์สำเร็จ {pages_count} หน้า!")
                
                

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
        with st.spinner("กำลังประมวลผลข้อมูลทั้งหมด..."):
            response = generate_complete_response(prompt, st.session_state.document_content)
            st.write(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})