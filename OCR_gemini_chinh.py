import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from docx import Document
import fitz  # PyMuPDF
import numpy as np

# Cấu hình Tesseract cho Windows
# Đảm bảo thay thế đường dẫn dưới đây bằng đường dẫn chính xác tới tesseract.exe trên máy của bạn
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r"D:\ungdung\Release-24.08.0-0\poppler-24.08.0\Library\bin"

def extract_text_from_docx(file_path):
    """Trích xuất văn bản từ file DOCX"""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf_text(file_path):
    """Trích xuất văn bản từ file PDF dạng text"""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def extract_text_from_pdf_image(file_path):
    """Trích xuất văn bản từ PDF dạng hình ảnh bằng Tesseract OCR"""
    text = ""
    results = []  # Lưu kết quả từng trang
    # Chuyển từng trang PDF thành ảnh
    images = convert_from_path(file_path, poppler_path=POPPLER_PATH, dpi=300)
    output_file = f"{os.path.splitext(file_path)[0]}_extracted.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, img in enumerate(images, 1):
            # OCR từng ảnh bằng Tesseract
            result = pytesseract.image_to_string(img, lang='vie')
            text += result + "\n"

            # Ghi kết quả vào file với đánh dấu số trang
            f.write(result)
            f.write("\n\n")
            f.flush()  # Đảm bảo ghi ngay lập tức
            print(f"✅ Đã xử lý và lưu trang {idx}")

    print(f"✅ Đã lưu toàn bộ nội dung vào: {output_file}")
    return text

def extract_text(file_path):
    """Xử lý trích xuất văn bản từ file DOCX hoặc PDF"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".pdf":
        # Trích xuất PDF dạng text trước
        text = extract_text_from_pdf_text(file_path)
        if text.strip():  # Nếu có văn bản
            return text
        else:
            # Nếu không có văn bản (dạng ảnh), dùng Tesseract OCR
            return extract_text_from_pdf_image(file_path)
    else:
        return "Định dạng file không được hỗ trợ"

def save_to_txt(original_path, content):
    """Lưu nội dung vào file .txt"""
    txt_path = os.path.splitext(original_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Đã lưu nội dung vào: {txt_path}")

# Ví dụ xử lý
file_path = "QD-100-Vv-Ban-hanh-quy-dinh-dao-tao-DH-theo-hoc-che-TC-2023.pdf"  # Thay bằng file bạn muốn xử lý
content = extract_text(file_path)
save_to_txt(file_path, content)
