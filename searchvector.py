import mysql.connector
import json
import faiss
import numpy as np
import requests


def embed_text_nomic(text):
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
            return embedding
        else:
            print(f"❌ Lỗi gọi Nomic API: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Lỗi khi kết nối Nomic: {e}")
        return None


def get_similar_vectors(query_text, top_k=7):
    try:
        # Load FAISS index
        index = faiss.read_index("nomic_index.faiss")

        # Get embedding for query text
        query_vector = embed_text_nomic(query_text)
        if not query_vector:
            return []

        # Convert to numpy array
        query_vector = np.array(query_vector).reshape(1, -1).astype('float32')

        # Search similar vectors
        distances, indices = index.search(query_vector, top_k)

        # Return vector IDs and their distances
        return list(zip(indices[0], distances[0]))

    except Exception as e:
        print(f"❌ Lỗi tìm kiếm vector: {str(e)}")
        return []


def get_text_content(vector_ids):
    try:
        # Connect to MySQL
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="khoaluan"
        )
        cursor = conn.cursor()

        results = []
        for idx, distance in vector_ids:
            if idx != -1:  # Skip invalid indices
                cursor.execute("SELECT id, noi_dung FROM data_ocr WHERE id = %s", (int(idx),))
                row = cursor.fetchone()
                if row:
                    similarity = 1 - distance  # Convert distance to similarity score
                    results.append({
                        'id': row[0],
                        'content': row[1],
                        'similarity': f"{similarity:.2%}"
                    })

        cursor.close()
        conn.close()
        return results

    except Exception as e:
        print(f"❌ Lỗi truy vấn MySQL: {str(e)}")
        return []


def main():
    query = input("Nhập câu cần tìm kiếm: ")

    # Get similar vector IDs
    similar_vectors = get_similar_vectors(query)
    if not similar_vectors:
        print("Không tìm thấy kết quả phù hợp")
        return

    # Get content for those vectors
    results = get_text_content(similar_vectors)

    print(f"\nKết quả tìm kiếm cho: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"Kết quả #{i}")
        print(f"ID: {result['id']}")
        print(f"Độ tương đồng: {result['similarity']}")
        print(f"Nội dung: {result['content']}")
        print("-" * 80)


if __name__ == "__main__":
    main()