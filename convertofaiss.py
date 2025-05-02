import mysql.connector
import json
import faiss
import numpy as np
from typing import Tuple
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

def update_faiss_index(faiss_path: str = "openai2_index.faiss") -> Tuple[bool, str]:
    """
    Update or create FAISS index with vectors from MySQL database where is_hoi_dap = 1
    Args:
        faiss_path: Path to save FAISS index file
    Returns:
        Tuple[bool, str]: (success status, message)
    """
    try:
        # Connect to MySQL
        conn = get_db_connection()
        cursor = conn.cursor()

        # FAISS index (OpenAI embedding dimension)
        d = 1536  # text-embedding-3-small dimension
        if os.path.exists(faiss_path):
            # Load existing index
            index = faiss.read_index(faiss_path)
            print("Loaded existing FAISS index")
        else:
            # Create new index
            index = faiss.IndexIDMap(faiss.IndexFlatL2(d))
            print("Created new FAISS index")

        # Load data from data_ocr table where is_hoi_dap = 1
        cursor.execute("SELECT id_ocr, vector FROM data_ocr WHERE is_hoi_dap = 1")
        results = cursor.fetchall()

        ids, vectors = [], []
        for row in results:
            try:
                vector = json.loads(row[1])
                if len(vector) == d:
                    vectors.append(np.array(vector, dtype=np.float32))
                    ids.append(row[0])
                else:
                    print(f"⚠️ Skipping vector id_ocr={row[0]} due to incorrect dimension")
            except Exception as e:
                print(f"❌ Error processing vector id_ocr={row[0]}: {str(e)}")

        # Add to FAISS
        if vectors:
            vectors_np = np.array(vectors, dtype=np.float32)
            ids_np = np.array(ids, dtype=np.int64)

            # Check for duplicate IDs
            existing_ids = set(faiss.vector_to_array(index.id_map))  # Sửa đổi ở đây
            new_vectors = []
            new_ids = []
            for vid, vector in zip(ids_np, vectors_np):
                if vid not in existing_ids:
                    new_vectors.append(vector)
                    new_ids.append(vid)


            if new_vectors:
                new_vectors_np = np.array(new_vectors, dtype=np.float32)
                new_ids_np = np.array(new_ids, dtype=np.int64)
                index.add_with_ids(new_vectors_np, new_ids_np)
                faiss.write_index(index, faiss_path)
                message = f"✅ Successfully updated FAISS index with {len(new_vectors)} new vectors!"
                success = True
            else:
                message = "ℹ️ No new vectors to add to FAISS index"
                success = True
        else:
            message = "❌ No valid vectors found in data_ocr with is_hoi_dap = 1"
            success = False

        # Close connection
        cursor.close()
        conn.close()
        print(message)
        return success, message

    except Exception as e:
        print(f"❌ Error updating FAISS index: {str(e)}")
        return False, f"❌ Error updating FAISS index: {str(e)}"

def load_faiss_index(faiss_path: str = "openai2_index.faiss") -> faiss.IndexIDMap:
    """
    Load FAISS index from file
    Args:
        faiss_path: Path to FAISS index file
    Returns:
        faiss.IndexIDMap: Loaded FAISS index
    """
    try:
        if os.path.exists(faiss_path):
            return faiss.read_index(faiss_path)
        else:
            print(f"❌ FAISS index file not found at {faiss_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading FAISS index: {str(e)}")
        return None
