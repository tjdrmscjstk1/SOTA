from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from predict import predict_inventory

app = Flask(__name__)
CORS(app)  # 모든 도메인 허용
# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 예측 API
@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files['image']
    result = predict_inventory(file)
    return jsonify(result)

# 에셋 이미지 제공
@app.route('/assets/<path:filename>')
def serve_asset(filename):
    return send_from_directory('static/assets', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)