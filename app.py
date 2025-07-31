from flask import Flask, render_template, request
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import os

app = Flask(__name__)

model_dir = os.path.join(os.path.dirname(__file__), 'distilbert_depression_model')
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)

# Label map aligns with your encoding
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
    app.run(debug=True)
