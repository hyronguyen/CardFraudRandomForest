from flask import Flask
from randomPredict import predict

app = Flask(__name__)

# Đăng ký route từ tệp predict.py
app.add_url_rule('/api/predict', 'predict', predict, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)
