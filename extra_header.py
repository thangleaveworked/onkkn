import os
import base64
import requests
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PyPDF2 import PdfReader, PdfWriter

# Thiết lập credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertex-453618-22aae8d2cbfd.json"


def init_vertex_ai():
    """Khởi tạo Vertex AI với thông tin xác thực."""
    vertexai.init(project="vertex-453618", location="us-central1")


def file_to_base64(file_path):
    """Đọc file và chuyển sang base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_first_and_last_page(pdf_path):
    """Trích xuất trang đầu và trang cuối của file PDF, lưu vào file tạm."""
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        # Lấy trang đầu (trang 0)
        if len(reader.pages) > 0:
            writer.add_page(reader.pages[0])

        # Lấy trang cuối
        if len(reader.pages) > 1:
            writer.add_page(reader.pages[-1])

        # Lưu vào file tạm
        temp_path = "temp_first_last.pdf"
        with open(temp_path, "wb") as f:
            writer.write(f)
        return temp_path
    except Exception as e:
        print(f"Error extracting first and last page: {str(e)}")
        return None


def process_base64_content(encoded_data):
    """Giải mã dữ liệu base64 và gửi đến Vertex AI."""
    document = Part.from_data(
        mime_type="application/pdf",
        data=base64.b64decode(encoded_data)
    )
    model = GenerativeModel("gemini-1.5-flash-002")
    chat = model.start_chat()
    prompt = ("""
        Dựa vào file sau trích xuất các nội dung theo yêu cầu bên dưới và trả về theo định dạng JSON.  
        {
            "so_van_ban": "số của văn bản",
            "ngay_ky": "(type:date)ngày ký văn bản",
            "nguoi_ky": "tên người ký văn bản",
            "chuc_vu": "chức vụ người ký",
            "tieu_de_van_ban": "tiêu đề của văn bản",
            "mo_ta": "nội dung mô tả tóm tắt văn bản",
            "thoi_gian_bd_hieu_luc": "(type:date)thời gian bắt đầu hiệu lực của văn bản",
            "thoi_gian_kt_hieu_luc": "(type:date)thời gian kết thúc hiệu lực văn bản",
            "co_quan_ban_hanh": "tên cơ quan ban hành văn bản",
            "loai_van_ban": (int)id loại văn bản t 1 tới 8 trong chú thích
        }
        Lưu ý: 
        các loại văn bản: 
        1: Báo cáo
        2: Biên bản
        3: Đề án
        4: Kế hoạch
        5: Quy chế
        6: Quyết định     
        7: Thông báo
        8: Văn bản nháp
        - Chỉ trả về thông tin dựa vào nội dung từ văn bản, không cần giải thích gì thêm.
        - Nếu không có thông tin nào thì trả về null.
        - Định dạng ngày theo chuẩn YYYY-MM-DD.
    """)
    response = chat.send_message([document, prompt])
    result = response.text.replace('```json', '').replace('```', '').strip()
    return result


def extract_content(data):
    """Trích xuất thông tin từ trang đầu và trang cuối của file PDF."""
    file_url = data.get('file_path')
    try:
        print(f"Downloading file from {file_url}")
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        file_path = "temp_file.pdf"
        with open(file_path, "wb") as f:
            f.write(response.content)

        # Trích xuất trang đầu và trang cuối
        temp_pdf_path = extract_first_and_last_page(file_path)
        if not temp_pdf_path:
            os.remove(file_path)
            return {"error": "Không thể trích xuất trang đầu và trang cuối."}

        # Chuyển file sang base64
        encoded_data = file_to_base64(temp_pdf_path)
        os.remove(file_path)
        os.remove(temp_pdf_path)

        # Khởi tạo Vertex AI
        init_vertex_ai()

        # Gửi file base64 lên Vertex AI
        result = process_base64_content(encoded_data)
        print(f"Extracted result: {result}")
        return result
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {"error": f"Lỗi xử lý: {str(e)}"}