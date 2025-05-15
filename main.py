from flask import Flask, request, jsonify, render_template
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define dataset paths
test_dir = 'unified_dataset/test'  # Replace with your testing dataset path

# Image preprocessing for testing
test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_data_gen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,  # Optimized batch size for evaluation
    class_mode='categorical'
)

# Limit test samples for faster validation
max_samples = 500  # Adjust this value based on your needs
x_subset, y_subset = [], []
for x_batch, y_batch in test_data:
    if len(x_subset) >= max_samples:
        break
    x_subset.extend(x_batch)
    y_subset.extend(y_batch)
x_subset = np.array(x_subset[:max_samples])
y_subset = np.array(y_subset[:max_samples])

# Load models
detection_model_path = "kidney_cancer_detection_model.h5"
grading_model_path = "kidney_cancer_modeldetection_model.h5"

if not os.path.exists(detection_model_path) or not os.path.exists(grading_model_path):
    raise FileNotFoundError("One or both model files are missing. Please train and save models first.")

detection_model = load_model(detection_model_path)
grading_model = load_model(grading_model_path)

# Evaluate detection model on subset
def evaluate_detection_model():
    loss, accuracy = detection_model.evaluate(x_subset, y_subset, verbose=1)
    print(f"Detection Model Test Loss: {loss}, Accuracy: {accuracy}")

    evaluate_detection_model()

# Get class indices and reverse mapping
class_indices = test_data.class_indices
detection_class_labels = {v: k for k, v in class_indices.items()}

def predict_image(filepath):
    img = load_img(filepath, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = detection_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    category = detection_class_labels[predicted_class]
    if confidence < 0.6:
        return "Prediction: Uncertain result, low confidence."
    elif category == "normal":
        return f"Prediction: No tumor detected. Confidence: {confidence * 100:.2f}%"
    else:
        return f"Prediction: Kidney cancer detected. Confidence: {confidence * 100:.2f}%"

# Get grading class labels
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
grade_class_labels = {v: k for k, v in test_generator.class_indices.items()}

def predict_image_grade(filepath):
    img = load_img(filepath, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = grading_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_grade = grade_class_labels[predicted_class]
    confidence = np.max(prediction) * 100

    return f"The predicted grade is: {predicted_grade} with a probability of {confidence:.2f}%"

@app.route("/")
def main():
    return render_template("main.html")

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/doc_ref')
def doc_ref():
    return render_template('doc_ref.html')

@app.route("/detect_grade")
def detect_grade():
    return render_template("detect_grade.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"message": "No selected file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}_{file.filename}")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(filepath)

    try:
        result = predict_image(filepath)
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"message": f"Error: {e}"}), 500
    finally:
        os.remove(filepath)

@app.route("/analyze_grade", methods=["POST"])
def analyze_grade():
    if "image" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"message": "No selected file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}_{file.filename}")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(filepath)

    try:
        result = predict_image_grade(filepath)
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"message": f"Error: {e}"}), 500
    finally:
        os.remove(filepath)

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=9874)
