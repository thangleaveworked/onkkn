import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
import docx
import nltk
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Kết nối MySQL
pool_config = {
    "pool_name": "mypool",
    "pool_size": 10,
    "host": os.getenv("MYSQL_HOST", "ec2-13-215-208-205.ap-southeast-1.compute.amazonaws.com"),
    "user": os.getenv("MYSQL_USER", "vanban"),
    "password": os.getenv("MYSQL_PASSWORD", "123456"),
    "database": os.getenv("MYSQL_DATABASE", "vanban"),
    "port": int(os.getenv("MYSQL_PORT", 3306))
}
db_pool = MySQLConnectionPool(**pool_config)

def get_db_connection():
    return db_pool.get_connection()

nltk.download("punkt")

def clean_text(text):
    """ Làm sạch văn bản """
    text = text.replace("<br>", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.replace("\n\n", "\n")
    return text

def split_text_with_context(text, window_size=6, overlap=1):
    """ Chia văn bản thành các đoạn nhỏ """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), window_size - overlap):
        chunk = " ".join(sentences[i:i + window_size])
        chunks.append(chunk)
    return chunks

def embed_text(text):
    """ Tạo embedding bằng OpenAI API """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        if embedding:
            print(f"Kích thước vector embedding: {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"❌ Lỗi khi tạo embedding: {e}")
        return None

def save_to_mysql(cursor, conn, text, vector, id_van_ban, is_hoi_dap):
    """Lưu văn bản và vector vào bảng data_ocr"""
    try:
        vector_json = json.dumps(vector)
        sql = """
        INSERT INTO data_ocr (noi_dung_ocr, vector, id_van_ban, is_hoi_dap)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql, (text, vector_json, id_van_ban, is_hoi_dap))
        conn.commit()
        print(f"✅ Đã lưu văn bản: {text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Lỗi lưu MySQL: {str(e)}")
        conn.rollback()
        return False

def read_file(file_path):
    """Đọc văn bản từ file TXT"""
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError("Chỉ hỗ trợ file .txt")
    except Exception as e:
        print(f"❌ Lỗi đọc file {file_path}: {str(e)}")
        raise


def process_text_to_mysql(file_path, id_van_ban, muc_do_bmat=0):
    """Xử lý file text, embed và lưu vào MySQL với id_van_ban và trich_xuat"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        text = read_file(file_path)
        text = clean_text(text)
        chunks = split_text_with_context(text)

        success_count = 0
        total_chunks = len(chunks)
        is_hoi_dap = 1 if muc_do_bmat == 1 else 0

        for chunk in chunks:
            vector = embed_text(chunk)
            if vector:
                if save_to_mysql(cursor, conn, chunk, vector, id_van_ban, is_hoi_dap):
                    success_count += 1

        cursor.close()
        conn.close()

        return {
            "total_chunks": total_chunks,
            "successful_embeddings": success_count,
            "status": "completed" if success_count > 0 else "failed"
        }
    except Exception as e:
        print(f"❌ Lỗi xử lý embedding: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

# if __name__ == "__main__":
#     try:
#         process_docx_to_mysql("QD DAOTAO 25_all_pages.txt")
#         print("✅ Xử lý hoàn tất!")
#     except Exception as e:
#         print(f"❌ Lỗi: {str(e)}")
#     finally:
#         cursor.close()
#         conn.close()