from flask import Flask
from flask_cors import CORS 
from randomPredict import predict

app = Flask(__name__)
CORS(app, origins=["http://localhost:3030"]) 

app.add_url_rule('/api/predict', 'predict', predict, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)
