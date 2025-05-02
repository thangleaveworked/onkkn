from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from chatbotopenai import main
from embeddingOpenai import process_text_to_mysql
from GeminiOCR import process_pdf_from_url
from convertofaiss import update_faiss_index
import mysql.connector
from functools import wraps
import jwt

import requests

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="khoaluan2"
    )

app = Flask(__name__)

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
        
        query = """
        SELECT id, cau_hoi, cau_tra_loi 
        FROM qa_bank 
        ORDER BY ngay_tao DESC 
        LIMIT 30
        """
        
        cursor.execute(query)
        qa_list = cursor.fetchall()
        
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
        url = "http://localhost:5005/webhooks/rest/webhook"
        payload = {
            "sender": "thang_user",
            "message": message,
        }

        try:
            response = requests.post(url, json=payload, timeout=5)  # timeout để tránh treo lâu
            rasa_response = response.json()
            print(rasa_response)

            if response.status_code == 200 and rasa_response:
                text_reply = rasa_response[0].get('text', '').strip().lower()
                if text_reply == 'pass':
                    # Rasa bảo 'pass' → gọi main xử lý nâng cao
                    result = main(message, history)
                    return jsonify({
                        'response': result['answer'],
                        'sources': result.get('references', [])
                    })
                else:
                    # Rasa trả lời đầy đủ → dùng luôn
                    return jsonify({
                        'response': rasa_response[0].get('text', ''),
                        'sources': []
                    })
        except Exception as e:
            print(f"Lỗi khi gọi Rasa: {e}")

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


@app.route('/api/ocr', methods=['POST'])
def ocr():
    try:
        data = request.json
        pdf_url = data.get('pdf_url')
        id_van_ban = data.get('id_van_ban')  # Get document ID from request

        if not pdf_url:
            return jsonify({
                'error': 'URL không hợp lệ'
            }), 400

        if not id_van_ban:
            return jsonify({
                'error': 'ID văn bản không hợp lệ'
            }), 400

        # Process PDF from URL
        result = process_pdf_from_url(pdf_url)

        if result and 'txt_path' in result:
            # Process the extracted text to create embeddings and save to database
            print(f"Bắt đầu xử lý embedding cho file {result['txt_path']} với ID văn bản {id_van_ban}")
            embedding_result = process_text_to_mysql(result['txt_path'], id_van_ban)
            print(f"Kết quả embedding: {embedding_result}")

            if update_faiss_index():

                return jsonify({
                    'success': True
                })
        else:
            return jsonify({
                'error': 'Không thể xử lý file PDF'
            }), 500

    except Exception as e:
        print(f"Error in OCR API: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)