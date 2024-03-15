from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# pred_bp = Blueprint('pred', __name__)
# Load the SavedModel
model = load_model('https://drive.google.com/drive/folders/1-3qIBo3Hm-FZVxlqYzLyiIrlsoURmgfG?usp=sharing')


# Define the route for prediction
# @app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    img = img.resize((224, 224))  # Resize to match model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Perform prediction
    predictions = model.predict(img_array)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    class_labels = ['BacHa', 'BachDongNu', 'CamThaoDat', 'CayOi', 'CoManTrau', 'CoMuc', 'DanhDanh', 'DauTam', 'DiaLien', 'DiepCa', 
    'DiepHaChau', 'DinhLang', 'DonLaDo', 'DuaCan', 'HungChanh', 'KeDauNgua', 'KimHoaThao', 
    'KimNgan', 'KimTienThao', 'LaLot', 'MaDe', 'NgaiCuu', 'NhanTran', 'PhenDen', 'RauMa', 'SaiDat', 'SoHuyet', 'TiaTo', 'TrinhNuHoangCung', 'VuSuaDat']
    predicted_class_label = class_labels[predicted_class_index]

    # Return the prediction result
    return jsonify({
        'class_label': predicted_class_label,
        'predictions': predictions.tolist()
    })

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)

