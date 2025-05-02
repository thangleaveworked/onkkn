import mysql.connector
import json
import numpy as np
import faiss
from datetime import datetime
from nomicTextToVector import embed_text_nomic, create_qa_faiss_index
import os
def get_db_connection():
    """Kết nối tới cơ sở dữ liệu MySQL"""
    return mysql.connector.connect(
        host="ec2-13-215-208-205.ap-southeast-1.compute.amazonaws.com",
        user="vanban",
        password="123456",
        database="vanban",
        port=3306
    )

def get_updated_qa_records():
    """Truy xuất các bản ghi trong qa_bank có ngay_cap_nhat là ngày hiện tại"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT id_qa, cau_hoi, is_hoi_dap, vector_cau_hoi
        FROM qa_bank
        WHERE DATE(ngay_cap_nhat) = CURRENT_DATE or vector_cau_hoi IS NULL
        """
        cursor.execute(query)
        records = cursor.fetchall()

        cursor.close()
        conn.close()
        return records
    except Exception as e:
        print(f"Error fetching updated QA records: {str(e)}")
        return []

def update_vector_in_db(id_qa, vector):
    """Cập nhật vector_cau_hoi trong bảng qa_bank"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Kiểm tra kiểu dữ liệu của vector
        if isinstance(vector, np.ndarray):
            vector_list = vector.tolist()
        elif isinstance(vector, list):
            vector_list = vector
        else:
            print(f"Invalid vector type for id_qa {id_qa}: {type(vector)}")
            return False

        vector_json = json.dumps(vector_list)
        query = "UPDATE qa_bank SET vector_cau_hoi = %s WHERE id_qa = %s"
        cursor.execute(query, (vector_json, id_qa))
        conn.commit()

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating vector for id_qa {id_qa}: {str(e)}")
        return False

def remove_vector_from_faiss(id_qa, faiss_file="qa_index.faiss"):
    """Xóa vector khỏi FAISS index dựa trên id_qa"""
    try:
        if not os.path.exists(faiss_file):
            print(f"FAISS index file {faiss_file} not found")
            return False

        index = faiss.read_index(faiss_file)
        if index.ntotal == 0:
            print("FAISS index is empty")
            return False

        # FAISS IndexIDMap sử dụng id_qa làm ID
        index.remove_ids(np.array([id_qa], dtype=np.int64))
        faiss.write_index(index, faiss_file)
        print(f"Removed vector for id_qa {id_qa} from FAISS index")
        return True
    except Exception as e:
        print(f"Error removing vector for id_qa {id_qa}: {str(e)}")
        return False

def update_qa_bank():
    """Xử lý các bản ghi qa_bank cập nhật trong ngày: xóa hoặc thêm vector vào FAISS"""
    try:
        records = get_updated_qa_records()
        if not records:
            print("No updated QA records found for today")
            return {"status": "no_records", "processed": 0}

        processed_count = 0
        faiss_file = "qa_index.faiss"

        for record in records:
            id_qa = record['id_qa']
            cau_hoi = record['cau_hoi']
            is_hoi_dap = record['is_hoi_dap']

            if is_hoi_dap == 0:
                # Xóa vector khỏi FAISS
                if remove_vector_from_faiss(id_qa, faiss_file):
                    print(f"Successfully removed vector for id_qa {id_qa}")
                    processed_count += 1
            elif is_hoi_dap == 1 and cau_hoi:
                # Embedding câu hỏi bằng Nomic
                vector = embed_text_nomic(cau_hoi)
                if vector is not None:
                    # Cập nhật vector vào cột vector_cau_hoi
                    if update_vector_in_db(id_qa, vector):
                        print(f"Updated vector for id_qa {id_qa} in database")
                        processed_count += 1
                    else:
                        print(f"Failed to update vector for id_qa {id_qa} in database")
                else:
                    print(f"Failed to generate embedding for id_qa {id_qa}")

        # Cập nhật FAISS index sau khi xử lý tất cả bản ghi
        if processed_count > 0:
            if create_qa_faiss_index(faiss_file):
                print("FAISS index updated successfully")
            else:
                print("Failed to update FAISS index")

        return {
            "status": "completed",
            "processed": processed_count,
            "total_records": len(records)
        }
    except Exception as e:
        print(f"Error  Error processing QA bank updates: {str(e)}")
        return {"status": "failed", "error": str(e)}
