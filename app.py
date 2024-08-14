import os
from flask import Flask, request, render_template

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename


model = load_model("keras_Model.h5", compile=False)
labels = open("labels.txt", "r").readlines()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


def predict(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    predictions = model.predict(data)
    index = np.argmax(predictions)
    class_name = labels[index]
    confidence_score = predictions[0][index]
    return class_name[2:], confidence_score


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            class_name, confidence_score = predict(image_path)

            return render_template("index.html", result=class_name, confidence_score=confidence_score, image_path=image_path)
    else:
        return render_template("index.html")
    
if __name__ == "__main__":
    app.run()