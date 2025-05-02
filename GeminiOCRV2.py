import os
import time
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PyPDF2 import PdfReader, PdfWriter
import os
from dotenv import load_dotenv
load_dotenv()
class PDFProcessingService:
    def __init__(self):
        self.credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "vertex-453618-22aae8d2cbfd.json")
        self.project_id = os.getenv("GOOGLE_PROJECT_ID", "vertex-453618")
        self.location = os.getenv("GOOGLE_LOCATION", "us-central1")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel("gemini-1.5-flash-002")

    def process_pdf(self, pdf_path: str, output_path: str, batch_size: int = 3) -> str:
        """Xử lý file PDF, trích xuất văn bản từ từng nhóm 3 trang, lưu vào file txt."""
        try:
            start_time = time.time()
            print(f"Bắt đầu xử lý file PDF: {pdf_path}")

            # Mở file PDF để lấy số trang
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            print(f"Tổng số trang: {total_pages}")

            # Mở file txt để ghi kết quả
            with open(output_path, "w", encoding="utf-8") as f:
                # Xử lý từng nhóm 3 trang
                for start_page in range(0, total_pages, batch_size):
                    # Tạo file PDF tạm cho nhóm 3 trang
                    temp_pdf_path = self.extract_page_batch(pdf_path, start_page, batch_size)
                    if not temp_pdf_path:
                        print(f"Không thể trích xuất trang {start_page + 1} đến {start_page + batch_size}.")
                        continue

                    # Chuyển file PDF tạm sang base64
                    encoded_data = self.file_to_base64(temp_pdf_path)

                    # Xóa file tạm
                    os.remove(temp_pdf_path)

                    # Gửi file base64 đến Vertex AI
                    result = self.process_base64_content(encoded_data)
                    if result:
                        # Ghi nội dung vào file txt
                        f.write(result + "\n\n")
                        f.flush()
                        print(f"Đã xử lý trang {start_page + 1} đến {min(start_page + batch_size, total_pages)}")
                    else:
                        print(f"Không nhận được kết quả từ Vertex AI cho trang {start_page + 1} đến {start_page + batch_size}.")

            duration = time.time() - start_time
            print(f"Hoàn thành xử lý trong {duration:.2f} giây")
            print(f"Kết quả được lưu tại: {output_path}")

            # Đọc nội dung file kết quả để trả về
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            return content if content.strip() else None

        except Exception as e:
            print(f"Lỗi xử lý PDF: {str(e)}")
            return None

    def extract_page_batch(self, pdf_path: str, start_page: int, batch_size: int) -> str:
        """Trích xuất một nhóm trang (batch_size) từ PDF, lưu vào file tạm."""
        try:
            reader = PdfReader(pdf_path)
            writer = PdfWriter()

            # Lấy các trang trong khoảng [start_page, start_page + batch_size)
            end_page = min(start_page + batch_size, len(reader.pages))
            for i in range(start_page, end_page):
                writer.add_page(reader.pages[i])

            # Lưu vào file tạm
            temp_path = f"temp_batch_{start_page}.pdf"
            with open(temp_path, "wb") as f:
                writer.write(f)
            return temp_path
        except Exception as e:
            print(f"Error extracting pages {start_page + 1} to {start_page + batch_size}: {str(e)}")
            return None

    def file_to_base64(self, file_path: str) -> str:
        """Đọc file và chuyển sang base64."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_base64_content(self, encoded_data: str) -> str:
        """Giải mã dữ liệu base64 và gửi đến Vertex AI với cơ chế retry."""
        max_retries = 5
        retry_delay = 7  # giây

        for attempt in range(1, max_retries + 1):
            try:
                document = Part.from_data(
                    mime_type="application/pdf",
                    data=base64.b64decode(encoded_data)
                )
                chat = self.model.start_chat()
                prompt = """
                Hãy trích xuất nội dung văn bản tiếng Việt từ file PDF:
                - Trích xuất toàn bộ nội dung text, sửa các lỗi định dạng hay chính tả nếu có.
                - Bỏ qua logo, hình ảnh trang trí.
                - Giữ lại thông tin người chữ ký và chức danh.
                - Trích xuất toàn bộ nội dung thành plaintext.
                - Nếu trang có bảng, định dạng theo markdown table.
                - Nếu nội dung là công thức toán học, chuyển thành cú pháp LaTeX và định dạng cho MathJax, bao quanh bằng $$ cho công thức trên dòng riêng hoặc $ cho công thức nội dòng, tùy ngữ cảnh.
                Chỉ trả về nội dung text đã trích xuất, không giải thích gì thêm.
                """
                response = chat.send_message([document, prompt])
                result = response.text.strip()
                print(f"Trích xuất thành công ở lần thử {attempt}")
                return result

            except Exception as e:
                print(f"Lỗi xử lý base64 content ở lần thử {attempt}: {str(e)}")
                if attempt < max_retries:
                    print(f"Thử lại sau {retry_delay} giây...")
                    time.sleep(retry_delay)
                else:
                    print(f"Đã thử {max_retries} lần, không thể trích xuất nội dung.")
                    return None

def process_pdf_from_url(pdf_path: str, txt_path: str) -> bool:
    """
    Xử lý PDF từ đường dẫn file và lưu văn bản đã trích xuất vào txt_path.
    Args:
        pdf_path: Đường dẫn tới file PDF
        txt_path: Đường dẫn tới file txt đích
    """
    try:
        # Khởi tạo service và xử lý PDF
        processor = PDFProcessingService()
        text_content = processor.process_pdf(pdf_path, txt_path, batch_size=3)
        return text_content is not None

    except Exception as e:
        print(f"Lỗi khi xử lý PDF: {e}")
        return False