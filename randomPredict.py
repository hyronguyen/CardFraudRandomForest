from flask import request, jsonify
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = 'model/rf_model_ori.joblib'
model = joblib.load(MODEL_PATH)

# Các feature yêu cầu mà mô hình yêu cầu
required_features = ['V1', 'V2', 'V3', 'V4', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'Amount']

def predict():
    try:
        data = request.get_json()

        features = data.get('features')

        if not features or not isinstance(features, list):
            return jsonify({'error': 'Missing or invalid features. It should be a list.'}), 400

        if len(features) != len(required_features):
            return jsonify({'error': f"Expected {len(required_features)} features, but got {len(features)}."}), 400

        features_df = pd.DataFrame([features], columns=required_features)

        prediction = model.predict(features_df)  
        probas = model.predict_proba(features_df)  

        gianLan = prediction[0].item() 
        xacXuat = probas[0][1].item()  
        ghiChu = "Dự đoán gian lận với xác suất cao" if gianLan == 1 and xacXuat > 0.8 else "Không gian lận" 

        return jsonify({
            'code': gianLan,
            'gianLan': gianLan,
            'xacXuatDuDoan': xacXuat,
            'ghiChu': ghiChu 
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500