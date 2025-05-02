from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from threading import Lock
from chatbotopenai import main
import textract
from apscheduler.schedulers.background import BackgroundScheduler
from docx import Document
from embeddingOpenai import process_text_to_mysql, embed_text
from GeminiOCRV2 import process_pdf_from_url
from datetime import datetime
from convertofaiss import update_faiss_index
from extra_header import extract_content
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from functools import wraps
import jwt
import os
import faiss
import numpy as np
from update_qa_bank import update_qa_bank, remove_vector_from_faiss
import json
# from PDFProcessingService import PDFProcessingService, save_extracted_text
import atexit
import requests
from dotenv import load_dotenv
load_dotenv()

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


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "5f214cacbd59c4e7e2925f6e5f214cacbd59c4e7")
faiss_lock = Lock()
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


@app.route('/')
@login_required
def home():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    # return render_template('index.html', user_email=session.get('user_email'))
    return render_template('index.html',
                           user_email=session.get('user_email'),
                           user_name=session.get('user_name'))  # Add user_name


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Call VLUTE SSO API
        url = 'https://sso.vlute.edu.vn/auth/realms/Dev/protocol/openid-connect/token'
        data = {
            'username': username,
            'password': password,
            'client_id': 'vlute.edu.vn',
            'client_secret': '',
            'grant_type': 'password',
            'scope': 'openid'
        }

        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data.get('access_token')
                print(access_token)
                # Decode JWT token
                decoded_token = jwt.decode(access_token, options={"verify_signature": False})
                user_email = decoded_token.get('email')
                user_name = decoded_token.get('name')
                print(user_name)

                # Store in session
                session['user_email'] = user_email
                session['user_name'] = user_name
                session['access_token'] = access_token

                return redirect(url_for('home'))
            else:
                return render_template('login.html', error='Invalid credentials')
        except Exception as e:
            print(f"Login error: {str(e)}")
            return render_template('login.html', error='Login failed')

    return render_template('login.html')


# Add at the end of the file
app.secret_key = '5f214cacbd59c4e7e2925f6e5f214cacbd59c4e7'  #


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/api/get-qa', methods=['GET'])
@login_required
def get_qa():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Lấy danh sách câu hỏi và câu trả lời
        query = """
        SELECT id_qa, cau_hoi, cau_tra_loi 
        FROM qa_bank 
        ORDER BY ngay_tao DESC 
        LIMIT 30
        """
        cursor.execute(query)
        qa_list = cursor.fetchall()

        # Lấy thông tin văn bản liên quan cho mỗi câu hỏi
        for qa in qa_list:
            qa['references'] = []
            cursor.execute("""
                SELECT v.tieu_de, v.ngay_ky, f.path_s3, v.ngay_bd_hieu_luc, v.ngay_het_hieu_luc, f.id_file
                FROM qa_bank_van_ban qbv
                JOIN files f ON qbv.id_file = f.id_file
                JOIN van_ban v ON f.id_van_ban = v.id_van_ban
                WHERE qbv.id_qa = %s
            """, (qa['id_qa'],))
            references = cursor.fetchall()
            for ref in references:
                qa['references'].append({
                    'name': ref['tieu_de'],
                    'date': ref['ngay_ky'].strftime('%Y-%m-%d') if ref['ngay_ky'] else 'N/A',
                    'path': "https://s3.vlute.edu.vn/vanban-vlute-edu-vn/"+ref['path_s3'],
                    'effective_date': ref['ngay_bd_hieu_luc'].strftime('%Y-%m-%d') if ref['ngay_bd_hieu_luc'] else 'N/A',
                    'expiry_date': ref['ngay_het_hieu_luc'].strftime('%Y-%m-%d') if ref['ngay_het_hieu_luc'] else 'N/A',
                    'id_file': ref['id_file']
                })

        cursor.close()
        conn.close()

        return jsonify(qa_list)
    except Exception as e:
        print(f"Error fetching QA data: {str(e)}")
        return jsonify([]), 500


@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.json
        message = data.get('message')
        history = data.get('history', [])

        if not message:
            return jsonify({
                'response': 'Không nhận được câu hỏi.',
                'sources': []
            })


        # Nếu có lỗi hoặc không vào các trường hợp trên → fallback sang main
        result = main(message, history)
        references = result.get('references', [])
        print(references)
        return jsonify({
            'response': result['answer'],
            'sources': result['references']  # Direct access since we know it exists
        })


    except Exception as e:
        print(f"Error in chat API: {str(e)}")
        return jsonify({
            'error': str(e),
            'sources': []
        }), 500


def get_new_files():
    """Truy xuất các file mới trong ngày từ bảng files"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT f.id_file, f.id_van_ban, f.path_s3, vb.muc_do_bmat, f.type
        FROM files f
        JOIN van_ban vb ON f.id_van_ban = vb.id_van_ban
        WHERE DATE(f.ngay_tao) = CURRENT_DATE AND f.ocr IS NULL;
        """
        cursor.execute(query)
        files = cursor.fetchall()

        cursor.close()
        conn.close()
        return files
    except Exception as e:
        print(f"Error fetching new files: {str(e)}")
        return []

def update_ocr_content(id_file, ocr_content):
    """Cập nhật nội dung OCR vào cột ocr của bảng files"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "UPDATE files SET ocr = %s WHERE id_file = %s"
        cursor.execute(query, (ocr_content, id_file))
        conn.commit()

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating OCR content: {str(e)}")
        return False



def save_text_to_file(text, txt_path):
    """Lưu văn bản vào file .txt"""
    try:

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return txt_path
    except Exception as e:
        print(f"Error saving text to file: {str(e)}")
        return None



def extract_text_from_docx(docx_path):
    """Trích xuất văn bản từ file .docx"""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text
    except Exception as e:
        print(f"Error extracting text from .docx: {str(e)}")
        return None




def extract_text_from_doc(doc_path):
    """Trích xuất văn bản từ file .doc (dùng textract)"""
    try:
        text = textract.process(doc_path)
        return text.decode('utf-8')
    except Exception as e:
        print(f"Error extracting text from .doc: {str(e)}")
        return None


@app.route('/api/search', methods=['POST'])
@login_required
def search_text():
    try:
        data = request.json
        query = data.get('query')

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Tạo embedding cho văn bản đầu vào
        query_embedding = embed_text(query)
        if query_embedding is None:
            return jsonify({"error": "Không thể tạo embedding cho văn bản đầu vào."}), 500

        # Tải chỉ mục FAISS
        faiss_index_path = "openai2_index.faiss"
        if not os.path.exists(faiss_index_path):
            return jsonify({"error": "Chỉ mục FAISS không tồn tại."}), 500

        index = faiss.read_index(faiss_index_path)
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)

        # Tìm kiếm k kết quả tương đồng nhất
        k = 5
        distances, indices = index.search(query_vector, k)

        # Lấy thông tin chi tiết từ cơ sở dữ liệu
        results = []
        with get_db_connection() as conn:
            for idx, score in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                idx = int(idx)  # Chuyển đổi numpy.int64 sang int

                # Tạo cursor mới cho mỗi truy vấn
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute("""
                        SELECT do.id_ocr, do.noi_dung_ocr, f.path_s3, v.tieu_de
                        FROM data_ocr do
                        JOIN files f ON do.id_van_ban = f.id_van_ban
                        JOIN van_ban v ON f.id_van_ban = v.id_van_ban
                        WHERE do.id_ocr = %s
                    """, (idx,))
                    result = cursor.fetchone()
                    cursor.fetchall()  # Đọc hết các hàng còn lại để tránh "Unread result"
                    if result:
                        results.append({
                            'text': result['noi_dung_ocr'],
                            'path': "https://s3.vlute.edu.vn/vanban-vlute-edu-vn/" + result['path_s3'],
                            'title': result['tieu_de'],
                            'score': float(1 / (1 + score))
                        })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/extract', methods=['POST'])
def extract_content_endpoint():
    """Trích xuất nội dung từ file PDF sử dụng extra_header.py."""
    data = request.get_json()
    if not data or 'file_path' not in data:
        return jsonify({"error": "Thiếu URL của file PDF."}), 400
    try:
        print(data)
        result = extract_content(data)
        # Kiểm tra nếu result là dictionary lỗi
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 500
        # Trả về result dưới dạng JSON
        return (result)
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


def update_data_ocr():
    """Xử lý các file mới: trích xuất văn bản (PDF hoặc DOC/DOCX), lưu nội dung, tạo embedding, cập nhật FAISS"""
    try:
        files = get_new_files()
        if not files:
            print("No new files found for today")
            return {"status": "no_files", "processed": 0}

        processed_count = 0
        os.makedirs("temp_files", exist_ok=True)
        os.makedirs("extracted_texts", exist_ok=True)

        for file in files:
            id_file = file['id_file']
            id_van_ban = file['id_van_ban']
            path_s3 = "https://s3.vlute.edu.vn/vanban-vlute-edu-vn/"+file['path_s3']
            file_type = file['type'].lower() if file['type'] else ''
            muc_do_bmat = file['muc_do_bmat']
            print(path_s3)
            # Tải file từ S3
            try:
                response = requests.get(path_s3)
                response.raise_for_status()

                # Lưu file tạm thời
                file_ext = '.pdf' if file_type == 'pdf' else '.docx' if file_type == 'docx' else '.doc'
                temp_filename = f"temp_{id_file}{file_ext}"
                temp_path = os.path.join("temp_files", temp_filename)
                with open(temp_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading file {path_s3}: {str(e)}")
                continue

            # Đường dẫn file .txt
            txt_path = os.path.join("extracted_texts", f"{id_file}.txt")
            content = None

            # Trích xuất văn bản
            if file_type == 'pdf':
                # Xử lý PDF và lưu vào txt_path
                result = process_pdf_from_url(temp_path, txt_path)
                if result:
                    # Đọc nội dung từ file .txt vừa tạo
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Error reading extracted text from {txt_path}: {str(e)}")
                        content = None
                else:
                    print(f"Failed to process PDF for file {id_file}")
            elif file_type == 'docx':
                # Trích xuất văn bản từ .docx
                content = extract_text_from_docx(temp_path)
                if content:
                    save_text_to_file(content, txt_path)
                else:
                    print(f"Failed to extract text from DOCX for file {id_file}")
            elif file_type == 'doc':
                # Trích xuất văn bản từ .doc
                content = extract_text_from_doc(temp_path)
                if content:
                    save_text_to_file(content, txt_path)
                else:
                    print(f"Failed to extract text from DOC for file {id_file}")
            else:
                print(f"Unsupported file type: {file_type}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                continue

            if content:
                # Cập nhật nội dung OCR vào bảng files
                if update_ocr_content(id_file, content):
                    print(f"Updated OCR content for file {id_file}")

                # Xử lý embedding và lưu vào data_ocr
                if(muc_do_bmat!=1):
                    muc_do_bmat=0
                else: muc_do_bmat=1
                embedding_result = process_text_to_mysql(txt_path, id_van_ban, muc_do_bmat)
                if embedding_result['status'] == 'completed':
                    print(f"Embedding completed for file {id_file}: {embedding_result}")

                # Cập nhật FAISS index
                if update_faiss_index():
                    print(f"FAISS index updated for file {id_file}")
                    processed_count += 1

            # Xóa file tạm
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(txt_path):
                os.remove(txt_path)  # Xóa file .txt sau khi xử lý

        return {
            "status": "completed",
            "processed": processed_count,
            "total_files": len(files)
        }
    except Exception as e:
        print(f"Error processing new files: {str(e)}")
        return {"status": "failed", "error": str(e)}


@app.route('/api/delete-file-vector', methods=['POST'])
def delete_file_vector():
    """Xóa các vector trong FAISS và bản ghi trong data_ocr dựa trên id_file"""
    try:
        data = request.json
        id_file = data.get('id_file')
        print(f"Received request to delete vectors and data_ocr for id_file: {id_file}")

        if not id_file or not isinstance(id_file, int):
            return jsonify({"error": "id_file không hợp lệ, phải là một số nguyên."}), 400

        # Kết nối cơ sở dữ liệu
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Tìm các bản ghi trong data_ocr với id_file
        cursor.execute("SELECT id_ocr FROM data_ocr WHERE id_file = %s", (id_file,))
        ocr_records = cursor.fetchall()

        if not ocr_records:
            print(f"No data_ocr records found for id_file: {id_file}")
            cursor.close()
            conn.close()
            return jsonify({"status": "success", "message": f"Không có vector hoặc bản ghi nào liên quan đến id_file {id_file}."})

        # Xóa vector trong FAISS
        faiss_file = "openai2_index.faiss"
        with faiss_lock:  # Sửa lỗi từ 'faologici' thành 'faiss_lock'
            if not os.path.exists(faiss_file):
                print(f"FAISS index not found at {faiss_file}")
                cursor.close()
                conn.close()
                return jsonify({"error": f"Chỉ mục FAISS không tồn tại tại {faiss_file}."}), 500

            index = faiss.read_index(faiss_file)
            ids_to_remove = [record['id_ocr'] for record in ocr_records]
            ids_np = np.array(ids_to_remove, dtype=np.int64)

            # Xóa vector khỏi FAISS
            index.remove_ids(ids_np)
            faiss.write_index(index, faiss_file)
            print(f"Removed {len(ids_to_remove)} vectors for id_file: {id_file}")

        # Xóa các bản ghi trong data_ocr
        cursor.execute("DELETE FROM data_ocr WHERE id_file = %s", (id_file,))
        conn.commit()
        deleted_count = cursor.rowcount
        print(f"Deleted {deleted_count} records from data_ocr for id_file: {id_file}")

        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "message": f"Đã xóa {len(ids_to_remove)} vector và {deleted_count} bản ghi data_ocr cho id_file {id_file}."
        })

    except Exception as e:
        print(f"Error in delete file vector API: {str(e)}")
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
        return jsonify({"error": str(e)}), 500


@app.route('/api/update-vanban-vectors', methods=['POST'])
def update_vanban_vectors():
    """Cập nhật vector trong FAISS dựa trên id_van_ban và muc_do_bmat"""
    try:
        data = request.json
        id_van_ban = data.get('id_van_ban')
        muc_do_bmat = data.get('muc_do_bmat')
        print(f"Nhận yêu cầu cập nhật vector cho id_van_ban: {id_van_ban}, muc_do_bmat: {muc_do_bmat}")

        if not id_van_ban or not isinstance(id_van_ban, int):
            error_msg = f"id_van_ban không hợp lệ, phải là số nguyên. Nhận được: {id_van_ban} (loại: {type(id_van_ban)})"
            print(error_msg)
            return jsonify({"error": error_msg}), 400
        if not isinstance(muc_do_bmat, int):
            error_msg = f"muc_do_bmat không hợp lệ, phải là số nguyên. Nhận được: {muc_do_bmat} (loại: {type(muc_do_bmat)})"
            print(error_msg)
            return jsonify({"error": error_msg}), 400

        # Kết nối cơ sở dữ liệu
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Xử lý theo muc_do_bmat
        faiss_file = "openai2_index.faiss"
        if muc_do_bmat == 1:
            # Thêm vector vào FAISS
            cursor.execute("""
                SELECT do.id_ocr, do.noi_dung_ocr, do.vector, f.path_s3
                FROM data_ocr do
                JOIN files f ON do.id_file = f.id_file
                WHERE do.id_van_ban = %s
            """, (id_van_ban,))
            ocr_records = cursor.fetchall()

            if not ocr_records:
                print(f"Không tìm thấy bản ghi data_ocr nào cho id_van_ban: {id_van_ban} với is_hoi_dap = 1")
                cursor.close()
                conn.close()
                return jsonify({
                    "status": "success",
                    "message": f"Không có bản ghi data_ocr nào để thêm vector cho id_van_ban {id_van_ban}."
                })

            with faiss_lock:
                if not os.path.exists(faiss_file):
                    print(f"Không tìm thấy chỉ mục FAISS tại: {faiss_file}")
                    cursor.close()
                    conn.close()
                    return jsonify({
                        "error": f"Chỉ mục FAISS không tồn tại tại {faiss_file}."
                    }), 500

                index = faiss.read_index(faiss_file)
                vectors = []
                ids = []

                for record in ocr_records:
                    if record['vector']:
                        try:
                            vector = np.array(json.loads(record['vector']), dtype=np.float32)
                            vectors.append(vector)
                            ids.append(record['id_ocr'])
                        except Exception as e:
                            print(f"Lỗi khi xử lý vector cho id_ocr {record['id_ocr']}: {str(e)}")
                            continue

                if vectors:
                    vectors_np = np.array(vectors, dtype=np.float32)
                    ids_np = np.array(ids, dtype=np.int64)
                    index.add_with_ids(vectors_np, ids_np)
                    faiss.write_index(index, faiss_file)
                    print(f"Đã thêm {len(vectors)} vector cho id_van_ban: {id_van_ban}")

                cursor.close()
                conn.close()

                return jsonify({
                    "status": "success",
                    "message": f"Đã thêm {len(vectors)} vector cho id_van_ban {id_van_ban}."
                })

        else:
            # Xóa vector trong FAISS
            cursor.execute("SELECT id_ocr FROM data_ocr WHERE id_van_ban = %s", (id_van_ban,))
            ocr_records = cursor.fetchall()

            if not ocr_records:
                print(f"Không tìm thấy bản ghi data_ocr nào cho id_van_ban: {id_van_ban}")
                cursor.close()
                conn.close()
                return jsonify({
                    "status": "success",
                    "message": f"Không có vector nào liên quan đến id_van_ban {id_van_ban}."
                })

            with faiss_lock:
                if not os.path.exists(faiss_file):
                    print(f"Không tìm thấy chỉ mục FAISS tại: {faiss_file}")
                    cursor.close()
                    conn.close()
                    return jsonify({
                        "error": f"Chỉ mục FAISS không tồn tại tại {faiss_file}."
                    }), 500

                index = faiss.read_index(faiss_file)
                ids_to_remove = [record['id_ocr'] for record in ocr_records]
                ids_np = np.array(ids_to_remove, dtype=np.int64)

                # Xóa vector khỏi FAISS
                index.remove_ids(ids_np)
                faiss.write_index(index, faiss_file)
                print(f"Đã xóa {len(ids_to_remove)} vector cho id_van_ban: {id_van_ban}")

            # Cập nhật is_hoi_dap = 0 trong data_ocr
            cursor.execute("UPDATE data_ocr SET is_hoi_dap = 0 WHERE id_van_ban = %s", (id_van_ban,))
            conn.commit()
            updated_count = cursor.rowcount
            print(f"Đã cập nhật {updated_count} bản ghi data_ocr cho id_van_ban: {id_van_ban}")

            cursor.close()
            conn.close()

            return jsonify({
                "status": "success",
                "message": f"Đã xóa {len(ids_to_remove)} vector và cập nhật {updated_count} bản ghi data_ocr cho id_van_ban {id_van_ban}."
            })

    except Exception as e:
        error_msg = f"Lỗi trong API cập nhật vector văn bản: {str(e)}"
        print(error_msg)
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
        return jsonify({"error": error_msg}), 500



@app.route('/api/delete-qa-vector', methods=['POST'])

def delete_qa_vector():
    """Xóa vector trong FAISS dựa trên id_qa"""
    try:
        data = request.json
        id_qa = data.get('id_qa')
        print(f"Received request to delete vector for id_qa: {id_qa}")

        if not id_qa or not isinstance(id_qa, int):
            return jsonify({"error": "id_qa không hợp lệ, phải là một số nguyên."}), 400

        # Gọi hàm xóa vector từ FAISS
        faiss_file = "qa_index.faiss"
        if remove_vector_from_faiss(id_qa, faiss_file):
            return jsonify({"status": "success", "message": f"Đã xóa vector cho id_qa {id_qa}."})
        else:
            return jsonify({"error": f"Không thể xóa vector cho id_qa {id_qa}. Kiểm tra file FAISS hoặc id_qa."}), 500

    except Exception as e:
        print(f"Error in delete QA vector API: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route('/api/ocr', methods=['POST'])
def ocr():
    """Endpoint để kích hoạt xử lý các file mới trong ngày"""
    try:
        result = update_data_ocr()
        return jsonify(result)
    except Exception as e:
        print(f"Error in OCR API: {str(e)}")
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/api/update-qa-bank', methods=['POST'])
def update_qa_bank_endpoint():
    """Endpoint để kích hoạt xử lý các bản ghi qa_bank cập nhật trong ngày"""
    try:
        result = update_qa_bank()
        return jsonify(result)
    except Exception as e:
        print(f"Error in update QA bank API: {str(e)}")
        return jsonify({"status": "failed", "error": str(e)}), 500

# Thêm scheduler để chạy tự động
# Lập lịch chạy tự động hàng ngày
scheduler = BackgroundScheduler()
scheduler.remove_all_jobs()
scheduler.add_job(func=update_qa_bank, trigger="cron", hour=16, minute=00)  # Chạy lúc 2:00 AM
scheduler.add_job(func=update_data_ocr, trigger="cron", hour=14, minute=00)  # Chạy lúc 1:00 AM
scheduler.start()

# Đảm bảo scheduler tắt khi ứng dụng dừng
if __name__ == '__main__':
    print("hello")
    if not scheduler.running:
        print("Starting scheduler")
        scheduler.start()
    app.run(debug=False,host="0.0.0.0", port=5000,threaded=True)

import atexit
atexit.register(lambda: scheduler.shutdown())