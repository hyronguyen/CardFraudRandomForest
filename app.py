from flask import Flask
from randomPredict import predict, predict2

app = Flask(__name__)

# Đăng ký route từ tệp predict.py
app.add_url_rule('/api/predict', 'predict', predict, methods=['POST'])
app.add_url_rule('/api/predict2', 'predict2', predict2, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)
