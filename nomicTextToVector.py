import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
import json
import requests
import numpy as np
import faiss
import datetime
from dotenv import load_dotenv
import os

pool_config = {
    "pool_name": "mypool",
    "pool_size": 20,
    "host": os.getenv("MYSQL_HOST", "ec2-13-215-208-205.ap-southeast-1.compute.amazonaws.com"),
    "user": os.getenv("MYSQL_USER", "vanban"),
    "password": os.getenv("MYSQL_PASSWORD", "123456"),
    "database": os.getenv("MYSQL_DATABASE", "vanban"),
    "port": int(os.getenv("MYSQL_PORT", 3306))
}
db_pool = MySQLConnectionPool(**pool_config)

def get_db_connection():
    return db_pool.get_connection()

def embed_text_nomic(text):
    """Gọi Nomic API để tạo embedding"""
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

def save_to_mysql(cau_hoi, vector, cau_tra_loi, file_ids=None):
    """Lưu câu hỏi, vector, câu trả lời vào qa_bank và file_ids vào qa_bank_van_ban"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Chuyển vector thành JSON
        vector_list = vector.tolist()
        vector_json = json.dumps(vector_list)
        now = datetime.datetime.now()
        # Lưu vào bảng qa_bank
        sql = "INSERT INTO qa_bank (cau_hoi, cau_tra_loi, vector_cau_hoi, is_hoi_dap,ngay_cap_nhat) VALUES (%s, %s, %s, %s)"
        val = (cau_hoi, cau_tra_loi, vector_json, 1, now)
        cursor.execute(sql, val)
        id_qa = cursor.lastrowid

        # Lưu file_ids vào qa_bank_van_ban
        if file_ids:
            for file_id in file_ids:
                cursor.execute("""
                    INSERT INTO qa_bank_van_ban (id_qa, id_file)
                    VALUES (%s, %s)
                """, (id_qa, file_id))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Đã lưu câu hỏi: {cau_hoi[:50]}... và câu trả lời: {cau_tra_loi[:50]}... vào MySQL")
        return id_qa

    except Exception as e:
        print(f"❌ Lỗi khi lưu vào database: {e}")
        return None

def create_qa_faiss_index(output_file="qa_index.faiss"):
    """Tạo FAISS index từ vectors trong bảng qa_bank"""
    try:
        # Kết nối CSDL
        conn = get_db_connection()
        cursor = conn.cursor()

        # Lấy tất cả vectors từ qa_bank
        cursor.execute("SELECT id_qa, vector_cau_hoi FROM qa_bank WHERE vector_cau_hoi IS NOT NULL")
        results = cursor.fetchall()

        if not results:
            print("❌ Không có dữ liệu vector trong qa_bank")
            cursor.close()
            conn.close()
            return False

        # Chuyển đổi vectors từ JSON string sang numpy array
        vectors = []
        ids = []
        for id_qa, vector_json in results:
            try:
                vector = np.array(json.loads(vector_json), dtype=np.float32)
                vectors.append(vector)
                ids.append(int(id_qa))
            except json.JSONDecodeError:
                print(f"❌ Lỗi decode JSON cho id_qa {id_qa}")
                continue

        vectors = np.array(vectors)
        ids = np.array(ids, dtype=np.int64)

        # Tạo và train FAISS index với IDMap
        dimension = vectors.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        faiss.normalize_L2(vectors)

        # Add vectors với IDs tương ứng
        index.add_with_ids(vectors, ids)

        # Lưu index
        faiss.write_index(index, output_file)
        print(f"✅ Đã tạo FAISS index với {len(vectors)} vectors và lưu vào {output_file}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Lỗi khi tạo FAISS index: {e}")
        cursor.close()
        conn.close()
        return False