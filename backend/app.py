from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from utils import predict_image  # Our prediction function from the Canvas

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from our React frontend

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Checks if a file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and prediction request."""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get both class and confidence from the predict function in utils.py
        predicted_class, confidence = predict_image(file_path)

        # Clean up the uploaded file after prediction
        os.remove(file_path)

        if predicted_class is not None:
            # Return both values in the JSON response
            return jsonify({'class': predicted_class, 'confidence': confidence})
        else:
            return jsonify({'error': 'Could not process image'}), 500

    return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    # Runs the app on port 5000, accessible from any IP address
    app.run(host='0.0.0.0', port=5000)

