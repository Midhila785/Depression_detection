from flask import Flask, render_template, request
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import os
import requests

app = Flask(__name__)

model_dir = os.path.join(os.path.dirname(__file__), 'distilbert_depression_model')
model_path = os.path.join(model_dir, 'tf_model.h5')

GOOGLE_DRIVE_FILE_ID = '1oKzbTF6AZA5z2Wd-nURmkhoj5OQJXKMF'


def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

# Download model if missing
if not os.path.exists(model_path):
    os.makedirs(model_dir, exist_ok=True)
    print("Downloading model file from Google Drive...")
    download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, model_path)
    print("Model download completed.")

# Load tokenizer and model AFTER model file is ensured
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)

label_map = {0: 'Highly Depressed', 1: 'Moderately Depressed', 2: 'Not Depressed'}

def predict_depression(text, max_length=64):
    inputs = tokenizer(
        text, return_tensors='tf', padding='max_length', truncation=True, max_length=max_length
    )
    outputs = model(**inputs)
    logits = outputs.logits[0].numpy()
    probs = tf.nn.softmax(logits).numpy()
    idx = probs.argmax()
    perc = probs[idx] * 100
    label = label_map[idx]
    return label, round(perc, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_text = request.form['text']
        level, perc = predict_depression(user_text)
        result = f"Predicted level: {level} ({perc}%)"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)
