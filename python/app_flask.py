from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
from ocr_engine import TamilOCR
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ocr = TamilOCR(
    model_path=r'D:\projects ai\tamil-handwritten-recognition\data\tamilnet_model.pt',
    char_map_path=r'D:\projects ai\tamil-handwritten-recognition\data\tamilchar.txt'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No file uploaded')
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    if file and file.filename.endswith('.bmp'):
        try:
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            print(f"Saving file to: {filepath}")  # Debug print
            file.save(filepath)
            if not os.path.exists(filepath):
                return render_template('index.html', error=f'Failed to save image to {filepath}')
            predicted_char = ocr.predict(filepath)
            print(f"Predicted character: {predicted_char}")  # Debug print
            return render_template('index.html', prediction=predicted_char, image=url_for('static', filename=f'uploads/{unique_filename}'))
        except Exception as e:
            print(f"Prediction error: {str(e)}")  # Debug print
            return render_template('index.html', error=f'Prediction error: {str(e)}')
    return render_template('index.html', error='Invalid file format. Please upload a .bmp file')

if __name__ == '__main__':
    app.run(debug=True)