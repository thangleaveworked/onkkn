import requests
import json


def get_embedding(text):
    url = "http://ollama:11434/api/embeddings"
    data = {
        "model": "nomic-embed-text",
        "prompt": text
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return response.text


text = "Đây là một đoạn văn bản cần embedding"
embedding = get_embedding(text)
print(embedding)
