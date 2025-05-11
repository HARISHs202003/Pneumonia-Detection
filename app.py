from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.utils import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf
import os
import re
import csv

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session handling

# Paths & Model
MODEL_PATH = 'models/Pneumonia_Model.h5'
UPLOAD_FOLDER = "./static/uploads/"
HEATMAP_FOLDER = "./static/heatmaps/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)

# CSV File Path
CSV_FILE_PATH = 'patient_predictions.csv'

# Ensure the CSV file exists and create it with appropriate headers if not
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient Name', 'DOB', 'Age', 'Phone Number', 'Image Filename', 'Prediction', 'Confidence (%)', 'Heatmap Path'])

# 🔹 Utility: Clean filenames
def clean_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

# 🔹 Utility: Allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 🔹 Generate Grad-CAM heatmap
def generate_gradcam(img_array, model, last_conv_layer="conv2d"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Assuming binary classification

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # Normalize

    return heatmap[0]

# 🔹 Overlay heatmap on image
def overlay_heatmap(img_path, model, filename):
    img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = generate_gradcam(img_array, model)

    # Read original image
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend images
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    heatmap_path = os.path.join(HEATMAP_FOLDER, f"heatmap_{filename}")
    cv2.imwrite(heatmap_path, superimposed_img)

    return heatmap_path

# 🔹 Store Patient Data and Prediction to CSV
def store_prediction_to_csv(patient_name, dob, age, phone, filename, prediction, confidence, heatmap_path):
    file_exists = os.path.exists(CSV_FILE_PATH)
    
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header only if the file doesn't already exist
        if not file_exists:
            writer.writerow(['Patient Name', 'DOB', 'Age', 'Phone Number', 'Image Filename', 'Prediction', 'Confidence (%)', 'Heatmap Path'])
        
        # Write the patient data and prediction information
        writer.writerow([patient_name, dob, age, phone, filename, prediction, confidence, heatmap_path])

# 🔹 Patient Info Page
@app.route('/', methods=['GET', 'POST'])
def patient_info():
    if request.method == 'POST':
        session['patient_name'] = request.form['patient_name'].strip()
        session['dob'] = request.form['dob'].strip()
        session['age'] = request.form['age'].strip()
        session['phone'] = request.form['phone'].strip()  # Capture phone number

        if not session['patient_name']:
            flash("Patient name is required!", "danger")
            return redirect(url_for('patient_info'))

        return redirect(url_for('upload'))
    return render_template('patient_info.html')

# 🔹 Upload & Prediction Page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'patient_name' not in session:
        flash("Please enter patient information first.", "warning")
        return redirect(url_for('patient_info'))

    if request.method == 'POST':
        imagefiles = request.files.getlist("imagefiles")
        patient_name = clean_filename(session['patient_name'])
        dob = session['dob']
        age = session['age']
        phone = session['phone']
        
        predictions = []
        for imagefile in imagefiles:
            if imagefile and allowed_file(imagefile.filename):
                filename = clean_filename(f"{patient_name}_{imagefile.filename}")
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                imagefile.save(image_path)

                # Process image
                img = load_img(image_path, target_size=(256, 256))
                x = img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)

                # Model Prediction
                classes = model.predict(x)
                prediction_score = float(classes[0][0])  # Fixed: Get confidence properly
                confidence = round(prediction_score * 100, 2)
                result = "Positive" if prediction_score >= 0.5 else "Negative"
                classification = f'{result} ({confidence}%)'

                # Generate heatmap
                heatmap_path = overlay_heatmap(image_path, model, filename)

                # Save to CSV
                store_prediction_to_csv(patient_name, dob, age, phone, filename, result, confidence, heatmap_path)

                predictions.append({
                    "imagePath": image_path,
                    "heatmapPath": heatmap_path,
                    "prediction": classification,
                    "confidence": confidence
                })
            else:
                flash("Invalid file format. Only PNG, JPG, and JPEG are allowed.", "danger")

        return render_template('upload.html', predictions=predictions, patient_name=session['patient_name'])

    return render_template('upload.html', patient_name=session['patient_name'])

# Run Flask App
if __name__ == '__main__':
    app.run(port=5000, debug=True)
