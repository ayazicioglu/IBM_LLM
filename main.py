import requests
import json
import os
from datetime import datetime
import time

start_time = datetime.now()
print(f"İşlem başlangıç zamanı: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

file_path = '1.json' #Bu dosya pdf dosyasinin jsonlara bolunmus hali olan dosyadir

if not os.path.exists(file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump([], file, ensure_ascii=False)

with open('train.json', 'r', encoding='utf-8') as file: # Bu dosya modelden gelen cevablari kaydedecegimiz dosyadir.
    data = json.load(file)

for i in range(1, len(data['paragraphs']) + 1):
    paragraph_content = next(
        p['content'] for p in data['paragraphs'] if p['paragraph_id'] == f'para_{i}'
    )

    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": paragraph_content
                }
            ]
        }
    )

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON hatası: {e}")
            print(f"Hatalı içerik:\n{content}")
            continue

        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                current_data = json.load(file)
            except json.JSONDecodeError:
                current_data = []

        if isinstance(parsed, list):
            current_data.extend(parsed)
        else:
            current_data.append(parsed)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(current_data, file, ensure_ascii=False, indent=2)

        print(f"Paragraf {i} için model cevabı kaydedildi.")
    else:
        print(f"HTTP isteği başarısız: {response.status_code}. Paragraf {i} için yanıt alınamadı.")

# Bitiş zamanını hesapla
end_time = datetime.now()
duration = end_time - start_time
duration_min = round(duration.total_seconds() / 60, 2)

print(f"\nİşlem bitiş zamanı: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Toplam süre: {duration_min} dakika")