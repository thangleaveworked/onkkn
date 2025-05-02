FROM python:3.10-slim

WORKDIR /app

# Copy requirements.txt và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo các thư mục cần thiết
RUN mkdir -p /app/temp_files /app/extracted_texts /app/faiss_data

# Copy mã nguồn (các file Python, JSON, v.v.)
COPY . .

# Thiết lập biến môi trường cho Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/vertex-453618-22aae8d2cbfd.json

# Chạy ứng dụng chính
CMD ["python", "main.py"]