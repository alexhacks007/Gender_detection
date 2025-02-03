from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO  # Import YOLO from the ultralytics package
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model_path = r'./Gender_detection.pt'  # Adjust this path as necessary

# Load YOLOv5 model using ultralytics library
model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('front.html', results=None)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']
    if image.filename == '':
        return redirect(request.url)

    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # Process the image and get results
    results = detect_gender(image_path)

    # Remove the image after processing to save space
    os.remove(image_path)

    return render_template('upload.html', image_path=image_path, results=results)

def detect_gender(image_path):
    image = cv2.imread(image_path)
    
    # Perform inference using YOLOv5 model
    results = model(image)  # results is a list of inference results

    men_count = 0
    women_count = 0

    for result in results:
        for detection in result.boxes:
            conf, cls = detection.conf, detection.cls  # Extract confidence and class
            if cls == 1:  # Assuming class 0 corresponds to 'person' (Men)
                men_count += 1
            elif cls == 0:  # Assuming class 1 corresponds to 'person' (Women)
                women_count += 1

    return {'men': men_count, 'women': women_count}

if __name__ == '__main__':
    app.run(debug=True)
