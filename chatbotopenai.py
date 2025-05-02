import numpy as np
import faiss
from mysql.connector.pooling import MySQLConnectionPool
import mysql.connector
from openai import OpenAI
import datetime
import re
from nomicTextToVector import save_to_mysql, create_qa_faiss_index
import requests
import json
from dotenv import load_dotenv
import os
load_dotenv()

# Khởi tạo OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Khởi tạo OpenAI client



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

def embed_text(text):
    """Chuyển câu truy vấn thành vector embedding"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Lỗi khi tạo embedding: {e}")
        return None


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
                return np.array(embedding, dtype=np.float32)
        else:
            print(f"❌ Lỗi gọi Nomic API: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Lỗi khi kết nối Nomic: {e}")
        return None


def get_extended_text(cursor, base_id):
    """Lấy đoạn văn bản chính và mở rộng 2 đoạn trước & 2 đoạn sau kèm thời gian"""
    result_texts = []
    file_paths = []
    unique_docs = {}
    current_date = datetime.datetime.now().date()

    for offset in range(-2, 3):
        cursor.execute("""
            SELECT d.noi_dung_ocr, v.so_van_ban, v.ngay_ky, v.tieu_de, 
                   f.path_s3, v.id_van_ban, v.ngay_bd_hieu_luc, v.ngay_het_hieu_luc, f.id_file
            FROM data_ocr d
            JOIN van_ban v ON d.id_van_ban = v.id_van_ban
            JOIN files f ON v.id_van_ban = f.id_van_ban
            WHERE d.id_ocr = %s
        """, (base_id + offset,))
        row = cursor.fetchone()
        # Đảm bảo làm sạch kết quả còn lại (nếu có)
        cursor.fetchall()  # Đọc hết các hàng còn lại để tránh "Unread result"
        if row:
            content = row[0]
            so_van_ban = row[1]
            ngay_ky = row[2]
            tieu_de = row[3]
            path_s3 = "https://s3.vlute.edu.vn/vanban-vlute-edu-vn/" + row[4]
            id_van_ban = row[5]
            ngay_bd_hieu_luc = row[6]
            ngay_het_hieu_luc = row[7]
            id_file = row[8]

            formatted_text = f"{content}\n[ID:{id_file}, NBDHL: {ngay_bd_hieu_luc}, NHHL: {ngay_het_hieu_luc}]"
            result_texts.append(formatted_text)

            doc_key = f"{id_van_ban}"
            if doc_key not in unique_docs:
                unique_docs[doc_key] = {
                    'name': tieu_de,
                    'date': ngay_ky,
                    'path': path_s3,
                    'id_file': id_file
                }

            file_paths.append({
                'name': tieu_de,
                'path': path_s3,
                'date': ngay_ky,
                'id_file': id_file
            })

    unique_docs_list = list(unique_docs.values())
    return "\n\n".join(result_texts), file_paths, unique_docs_list


def search_faiss(query, top_k=5, faiss_file="openai2_index.faiss"):
    """Tìm kiếm văn bản gần đúng nhất từ FAISS và lấy nội dung từ MySQL"""
    try:
        results = []
        all_unique_docs = {}
        index = faiss.read_index(faiss_file)
        if index.ntotal == 0:
            print("FAISS index trống! Kiểm tra dữ liệu.")
            return [], []

        query_vector = embed_text(query)
        if query_vector is None:
            return [], []

        query_vector = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, top_k)

        conn = get_db_connection()
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            # Tạo cursor mới cho mỗi lần gọi get_extended_text
            cursor = conn.cursor()
            extended_text, file_paths, unique_docs = get_extended_text(cursor, int(idx))
            cursor.close()  # Đóng cursor sau khi sử dụng

            results.append((extended_text, distances[0][i]))

            print(f"\nĐoạn văn bản {i + 1} (id_ocr: {idx}):")
            print(f"- Độ tương đồng: {distances[0][i]:.2%}")
            print("-" * 50)
            print(extended_text)
            print("-" * 50)

            for doc in unique_docs:
                doc_key = f"{doc['name']}_{doc['date']}"
                if doc_key not in all_unique_docs:
                    all_unique_docs[doc_key] = doc

        conn.close()  # Đóng kết nối sau khi sử dụng

        print("\n📑 Danh sách văn bản độc nhất:")
        for doc in all_unique_docs.values():
            print(f"- {doc['name']} (ban hành ngày {doc['date']})")

        return results, list(all_unique_docs.values())
    except Exception as e:
        print(f"Lỗi khi tìm kiếm FAISS: {e}")
        return [], []


def search_qa_faiss(query_vector, top_k=3, faiss_file="qa_index.faiss"):
    """Tìm kiếm vector tương đồng từ qa_index.faiss"""
    try:
        index = faiss.read_index(faiss_file)
        if index.ntotal == 0:
            print("❌ FAISS index trống!")
            return []

        query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                similarity = float(1 - distances[0][i])
                results.append({
                    'id': int(idx),
                    'similarity': similarity
                })

        return results
    except Exception as e:
        print(f"❌ Lỗi khi tìm kiếm vector tương đồng: {e}")
        return []


def generate_chatgpt_response(query, context_texts):
    """Gửi dữ liệu cho ChatGPT để tạo câu trả lời"""
    try:
        context = "\n\n".join(context_texts)
        prompt = f"Người dùng hỏi: {query}\n\nDựa vào thông tin sau:\n{context}\n\nHãy trả lời câu hỏi một cách ngắn gọn, rõ ràng, chỉ sử dụng thông tin cung cấp. Nếu không có thông tin phù hợp, trả lời: 'Không tìm thấy thông tin phù hợp.'"

        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia tư vấn kỹ thuật trồng ớt sừng trâu."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Lỗi khi gọi ChatGPT: {e}")
        return "Xin lỗi, tôi không thể tạo câu trả lời vào lúc này."


def format_math_response(response):
    """Format lại công thức toán học để hiển thị với MathJax"""
    response = response.replace('<p>', '').replace('</p>', '')
    response = response.replace('\\[', '[').replace('\\]', ']')
    if not (response.startswith('$$') or response.startswith('\\[')):
        response = '$$' + response + '$$'
    return response


def analyze_chat_history_with_ai(query, chat_history):
    """Sử dụng OpenAI để phân tích lịch sử chat và tạo các câu truy vấn liên quan"""
    try:
        if not chat_history:
            return [query]

        formatted_history = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: " +
            (msg['content'][0]['text'] if isinstance(msg['content'], list) else msg['content'])
            for msg in chat_history[-6:]
        ])
        current_date = datetime.datetime.now().strftime("%d/%m/%Y")

        analysis_prompt = f"""Dựa vào lịch sử trò chuyện sau và câu hỏi hiện tại, hãy tạo ra 1-2 câu truy vấn cụ thể để tìm kiếm thông tin liên quan:
                        Lịch sử trò chuyện:
                        {formatted_history}

                        Câu hỏi hiện tại: {query}
                        Thời gian hiện tại: {current_date}

                        Yêu cầu:
                        - Nếu không đề cập đến thời gian, thêm yếu tố thời gian mới nhất.
                        - Mỗi câu truy vấn ngắn gọn, tập trung vào một khía cạnh cụ thể.
                        - Tận dụng ngữ cảnh từ lịch sử trò chuyện.
                        - Trả về dưới dạng danh sách, mỗi dòng một câu truy vấn."""

        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",  # Cập nhật model nếu cần
            messages=[
                {"role": "system",
                 "content": "Bạn là chuyên gia phân tích ngữ cảnh và tạo câu truy vấn trong hỏi đáp các nội dung văn bản giáo dục của trường Đại học Sư Phạm Kỹ Thuật Vĩnh Long."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.6,
            max_tokens=600
        )

        generated_queries = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
        generated_queries = [q.replace('- ', '') for q in generated_queries]
        all_queries = [query] + generated_queries
        return list(set(all_queries))
    except Exception as e:
        print(f"Lỗi khi phân tích lịch sử chat với AI: {e}")
        return [query]


def main(query, chat_history=None):
    """Xử lý câu truy vấn và trả về câu trả lời"""
    try:
        current_date = datetime.datetime.now().strftime("%d/%m/%Y")
        queries = analyze_chat_history_with_ai(query, chat_history)
        print(f"\n🔍 Các câu truy vấn được tạo ra:\n{queries}")

        best_match_data = None
        highest_similarity = 0
        best_match_id = None

        for q in queries:
            embedding = embed_text_nomic(q)
            if embedding is not None:
                similar_results = search_qa_faiss(embedding)
                print(similar_results)
                if similar_results:
                    print(f"\n📌 Câu hỏi tương tự cho '{q}':")
                    for result in similar_results:
                        current_similarity = result['similarity']
                        print(f"ID: {result['id']}, Độ tương đồng: {current_similarity:.2%}")
                        if current_similarity > 0.98 and current_similarity > highest_similarity:
                            highest_similarity = current_similarity
                            best_match_id = result['id']

        if best_match_id is not None:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cau_hoi, cau_tra_loi
                FROM qa_bank 
                WHERE id_qa = %s
            """, (best_match_id,))
            qa_data = cursor.fetchone()
            if qa_data:
                best_match_data = {
                    'answer': qa_data[1],
                    'doc_ids': []  # Sẽ lấy từ qa_bank_van_ban
                }
                print(f"\n✅ Tìm thấy câu trả lời phù hợp (độ tương đồng: {highest_similarity:.2%})")
                print(f"- Q: {qa_data[0]}")

                # Lấy id_file từ qa_bank_van_ban
                cursor.execute("""
                    SELECT id_file
                    FROM qa_bank_van_ban
                    WHERE id_qa = %s
                """, (best_match_id,))
                file_ids = [row[0] for row in cursor.fetchall()]
                best_match_data['doc_ids'] = file_ids

            cursor.close()
            conn.close()

        if best_match_data:
            conn = get_db_connection()
            cursor = conn.cursor()
            references = []
            for file_id in best_match_data['doc_ids']:
                cursor.execute("""
                    SELECT v.tieu_de, v.ngay_ky, f.path_s3, v.ngay_bd_hieu_luc, v.ngay_het_hieu_luc
                    FROM files f
                    JOIN van_ban v ON f.id_van_ban = v.id_van_ban
                    WHERE f.id_file = %s
                """, (file_id,))
                result = cursor.fetchone()
                if result:
                    references.append({
                        'name': result[0],
                        'date': result[1],
                        'path': "https://s3.vlute.edu.vn/vanban-vlute-edu-vn/"+result[2],
                        'effective_date': result[3],
                        'expiry_date': result[4]
                    })
            cursor.close()
            conn.close()

            chat_response = best_match_data['answer']
            if references:
                reference_section = "\n\n<div class='reference-section' style='background-color: #4e8b7c45; padding: 2px; border-radius: 5px;'>\nCăn cứ văn bản:\n<ul>"
                for ref in references:
                    effective_date = ref['effective_date'] if ref['effective_date'] else 'N/A'
                    expiry_date = ref['expiry_date'] if ref[
                        'expiry_date'] else None  # Đặt None nếu không có expiry_date
                    date_info = f"(ban hành ngày {ref['date']}, hiệu lực từ {effective_date})"
                    if expiry_date:  # Chỉ thêm phần "đến" nếu expiry_date tồn tại
                        date_info = f"(ban hành ngày {ref['date']}, hiệu lực từ {effective_date} đến {expiry_date})"

                    reference_section += f"""
                        <li style="margin-bottom: 3px; list-style: none;">
                            <a href="{ref['path']}" target="_blank" style="color: #4b6fe0; text-decoration: none; font-weight: 600; font-size: large;">{ref['name']}</a>
                            <span style="margin-left: 2px; font-size: 0.9em;">{date_info}</span>
                        </li>"""
                reference_section += "\n</ul>\n</div>"
                chat_response += reference_section

            return {
                'answer': chat_response,
                'references': references
            }

        formatted_queries = "\n".join([f"- {q}" for q in queries[1:]])

        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": f"""Bạn là một trợ lý thông minh trả lời các câu hỏi hỏi đáp các nội dung văn bản giáo dục của trường Đại học Sư Phạm Kỹ Thuật Vĩnh Long cho giảng viên và sinh viên.
                    # YÊU CẦU
                    - Chỉ sử dụng thông tin từ các đoạn văn bản được cung cấp.
                    - Chọn các thông tin phù hợp với thời gian hiệu lực (ngay_bd_hieu_luc <= hôm nay <= ngay_het_hieu_luc).
                    - Không tự suy diễn hoặc thêm thông tin ngoài văn bản.
                    - Câu trả lời rõ ràng, đúng trọng tâm, dễ hiểu.
                    # PHÂN TÍCH CÂU HỎI
                    Câu hỏi gốc: {query}
                    Các khía cạnh cần trả lời:
                    {formatted_queries}
                    # THỜI GIAN HỆ THỐNG
                    - Ngày hiện tại: {current_date}
                    - Chỉ sử dụng văn bản còn hiệu lực theo ngày hiện tại.
                    # CHÚ THÍCH
                    ID là id_van_ban, NBDHL là ngày bắt đầu hiệu lực, NHHL là ngày hết hiệu lực.
                    # XỬ LÝ KHI THIẾU THÔNG TIN
                    - Nếu không có văn bản phù hợp, trả lời: "**Không tìm thấy thông tin phù hợp để trả lời câu hỏi. Vui lòng đặt câu hỏi cụ thể hơn.**"
                    # ĐỊNH DẠNG TRẢ LỜI
                    - Trả lời bằng tiếng Việt, không giải thích thêm.
                    - ĐẶC BIỆT: Kết thúc câu trả lời, liệt kê ra id_file của các văn bản được sử dụng trong cặp dấu ## ##, ví dụ: ## 1,2 ## """
                }]
            }
        ]

        if chat_history and len(chat_history) > 0:
            messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": query
            }]
        })

        all_search_results = []
        all_unique_docs = {}
        for q in queries[1:]:
            results, unique_docs = search_faiss(q, top_k=2)
            if results:
                all_search_results.extend(results)
                for doc in unique_docs:
                    doc_key = f"{doc['name']}_{doc['date']}"
                    if doc_key not in all_unique_docs:
                        all_unique_docs[doc_key] = doc

        if all_search_results:
            context_texts = []
            file_ids = set()
            for text, _ in all_search_results:
                if isinstance(text, tuple):
                    content, _ = text
                    context_texts.append(content)
                else:
                    context_texts.append(text)

            context = "\n\n".join(context_texts)
            references = list(all_unique_docs.values())
            file_ids = {doc['id_file'] for doc in references if 'id_file' in doc}

            if references:
                reference_section = "Căn cứ văn bản:"
                for ref in references:
                    reference_section += f"- {ref['name']} (ban hành ngày {ref['date']})\n"
                context += reference_section

            messages.append({
                "role": "system",
                "content": [{
                    "type": "file_search_result",
                    "text": context
                }]
            })
        else:
            print("❌ Không tìm thấy kết quả tìm kiếm phù hợp.")
            references = []
            file_ids = set()

        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{
                "role": msg["role"],
                "content": msg["content"][0]["text"] if isinstance(msg["content"], list) else msg["content"]
            } for msg in messages],
            temperature=0.4,
            max_tokens=8096
        )

        chat_response = response.choices[0].message.content.strip()
#         chat_response = """Công thức tính điểm trung bình học kỳ (ĐTBHK) và điểm trung bình chung tích lũy (ĐTBCTL) được quy định như sau:
#
# $$A = \\frac{\sum_{i=1}^{n} a_i \\times n_i}{\sum_{i=1}^{n} n_i}$$
#
# Trong đó:
# - \(A\) là điểm trung bình chung HK hoặc điểm trung bình chung tích lũy.
# - \(a_i\) là điểm của học phần thứ \(i\).
# - \(n_i\) là số tín chỉ của học phần thứ \(i\).
# - \(n\) là tổng số học phần.
#
# Điểm trung bình chung HK được dùng để xét học bổng, khen thưởng sau mỗi học kỳ và để xếp loại học lực sinh viên.
#
# ## 49 ##"""
        print(chat_response)
        matches = re.findall(r'##\s*([\d,\s]+)\s*##', chat_response)
        doc_ids = [id.strip() for id in matches[0].split(',')] if matches else []
        print(doc_ids)

        chat_response = re.sub(r'##\s*[\d,\s]+\s*##', '', chat_response).rstrip()

        conn = get_db_connection()
        cursor = conn.cursor()
        references = []
        for doc_id in doc_ids:
            cursor.execute("""
                SELECT v.tieu_de, v.ngay_ky, f.path_s3, v.ngay_bd_hieu_luc, v.ngay_het_hieu_luc, f.id_file
                FROM files f
                JOIN van_ban v ON f.id_van_ban = v.id_van_ban
                WHERE f.id_file = %s
            """, (doc_id,))
            result = cursor.fetchone()
            if result:
                references.append({
                    'name': result[0],
                    'date': result[1],
                    'path': "https://s3.vlute.edu.vn/vanban-vlute-edu-vn/"+result[2],
                    'effective_date': result[3],
                    'expiry_date': result[4],
                    'id_file': result[5]
                })

        if chat_response != "**Không tìm thấy thông tin phù hợp để trả lời câu hỏi. Vui lòng đặt câu hỏi cụ thể hơn.**":
            for q in queries:
                if q!= query:
                    vector = embed_text_nomic(q)
                    if vector is not None:
                        cursor.execute("""
                            INSERT INTO qa_bank (cau_hoi, cau_tra_loi, is_hoi_dap, vector_cau_hoi)
                            VALUES (%s, %s, %s, %s)
                        """, (q, chat_response, 1, json.dumps(vector.tolist())))
                        id_qa = cursor.lastrowid

                        for file_id in file_ids:
                            cursor.execute("""
                                INSERT INTO qa_bank_van_ban (id_qa, id_file)
                                VALUES (%s, %s)
                            """, (id_qa, file_id))

                        conn.commit()
                        create_qa_faiss_index()

        reference_section = ""
        if references:
            reference_section = "\n\n<div class='reference-section' style='background-color: #4e8b7c45; padding: 2px; border-radius: 5px;'>\nCăn cứ văn bản:\n<ul>"
            for ref in references:
                reference_section += f"""
                    <li style="margin-bottom: 3px; list-style: none;">
                        <a href="{ref['path']}" target="_blank" style="color: #4b6fe0; text-decoration: none; font-weight: 600; font-size: large;">{ref['name']}</a>
                        <span style="color: #ffff; margin-left: 2px; font-size: 0.9em;">(ban hành ngày {ref['date']}, hiệu lực từ {ref['effective_date']} đến {ref['expiry_date']})</span>
                    </li>"""
            reference_section += "\n</ul>\n</div>"
            chat_response += reference_section

        cursor.close()
        conn.close()
        print(references)

        return {
            'answer': chat_response,
            'references': references
        }

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        return {
            'answer': "Xin lỗi, hiện tại tôi không thể xử lý câu hỏi của bạn.",
            'references': []
        }