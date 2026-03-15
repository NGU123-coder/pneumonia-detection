from flask import Flask, render_template, request
import os
from predict import predict_pneumonia
from gradcam import generate_gradcam

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
HEATMAP_FOLDER = "static/heatmaps"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)


@app.route("/", methods=["GET","POST"])
def index():

    result = None
    confidence = None
    heatmap_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file:

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)

            file.save(filepath)

            result, confidence = predict_pneumonia(filepath)

            heatmap_file = "heatmap_" + file.filename
            heatmap_output = os.path.join(HEATMAP_FOLDER, heatmap_file)

            generate_gradcam(filepath, heatmap_output)

            heatmap_path = heatmap_output

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        heatmap_path=heatmap_path
    )


if __name__ == "__main__":
    app.run(debug=True)