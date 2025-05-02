import mysql.connector
import docx
import nltk
import re
import json
import requests

# Kết nối MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Mặc định XAMPP không có mật khẩu
    database="khoaluan"
)
cursor = conn.cursor()

# Tải tokenizer NLTK
nltk.download("punkt")

# def read_docx(file_path):
#     """ Đọc văn bản từ file DOCX """
#     doc = docx.Document(file_path)
#     text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
#     return text

def clean_text(text):
    """ Làm sạch văn bản """
    # Xóa các thẻ <br> và khoảng trắng thừa
    text = text.replace("<br>", " ")

    # Xóa nhiều khoảng trắng liên tiếp
    text = re.sub(r"\s+", " ", text)

    # Xóa khoảng trắng đầu và cuối
    text = text.strip()

    # Chuẩn hóa xuống dòng
    text = text.replace("\n\n", "\n")

    return text

def split_text_with_context(text, window_size=2, overlap=1):
    """ Chia văn bản thành các đoạn nhỏ """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), window_size - overlap):
        chunk = " ".join(sentences[i:i + window_size])
        chunks.append(chunk)
    return chunks

def embed_text_nomic(text):
    """ Gọi Nomic API để tạo embedding """
    try:
        url = "http://ollama:11434/api/embeddings"
        data = {
            "model": "nomic-embed-text",
            "prompt": text
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            embedding = response.json().get("embedding")
            if embedding and isinstance(embedding, list):
                print(f"Kích thước vector embedding: {len(embedding)}")
            return embedding
        else:
            print(f"❌ Lỗi gọi Nomic API: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Lỗi khi kết nối Nomic: {e}")
        return None

def save_to_mysql(text, vector):
    """ Lưu văn bản và vector vào MySQL """
    vector_json = json.dumps(vector)  # Chuyển thành chuỗi JSON
    sql = "INSERT INTO data_ocr (noi_dung, vector, id_van_ban) VALUES (%s, %s, 1)"
    cursor.execute(sql, (text, vector_json))
    conn.commit()
    print(f"✅ Đã lưu văn bản: {text[:50]}...")

def read_file(file_path):
    """Đọc văn bản từ file DOCX hoặc TXT"""
    if file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format. Only .docx and .txt files are supported.")

def process_docx_to_mysql(file_path):
    """ Xử lý DOCX, embed và lưu vào MySQL """
    text = read_file(file_path)
    text = clean_text(text)
    chunks = split_text_with_context(text)

    for chunk in chunks:
        vector = embed_text_nomic(chunk)
        # if vector:
        #     print(f"Embedding cho đoạn: {chunk[:50]}...")
        #     save_to_mysql(chunk, vector)

# Chạy chương trình
if __name__ == "__main__":
    process_docx_to_mysql("QD DAOTAO 25_all_pages.txt")

# Đóng kết nối MySQL
cursor.close()
conn.close()
