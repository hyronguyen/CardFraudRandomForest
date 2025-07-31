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
        # Lấy dữ liệu JSON từ request
        data = request.get_json()

        # Lấy giá trị từ 'features'
        features = data.get('features')

        # Kiểm tra tính hợp lệ của features
        if not features or not isinstance(features, list):
            return jsonify({'error': 'Missing or invalid features. It should be a list.'}), 400

        # Kiểm tra xem số lượng features có đúng không
        if len(features) != len(required_features):
            return jsonify({'error': f"Expected {len(required_features)} features, but got {len(features)}."}), 400

        # Chuyển đổi features thành pandas DataFrame với tên cột đúng
        features_df = pd.DataFrame([features], columns=required_features)

        # Dự đoán với mô hình
        prediction = model.predict(features_df)  # Dự đoán nhãn (class)
        probas = model.predict_proba(features_df)  # Dự đoán xác suất cho mỗi lớp

        # Nhãn dự đoán (0 hoặc 1)
        gianLan = prediction[0].item()  # Chuyển int64 thành int

        # Xác suất của lớp "1" (gian lận)
        xacXuat = probas[0][1].item()  # Chuyển int64 thành int

        # Trả về kết quả dưới dạng JSON với cấu trúc yêu cầu
        return jsonify({
            'code': gianLan,
            'gianLan': gianLan,
            'xacXuatDuDoan': xacXuat
        })

    except Exception as e:
        # In lỗi ra log để dễ dàng theo dõi
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500