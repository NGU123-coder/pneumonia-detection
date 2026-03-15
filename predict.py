import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from lung_segmentation import segment_lungs

model = load_model("model/pneumonia_resnet50.h5")

def predict_pneumonia(img_path):

    # segment lungs first
    segmented = segment_lungs(img_path)

    resized = cv2.resize(segmented,(224,224))

    img_array = resized/255.0

    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)

    prob = prediction[0][0]

    if prob >= 0.5:
        result = "PNEUMONIA DETECTED"
        confidence = round(prob*100,2)
    else:
        result = "NORMAL"
        confidence = round((1-prob)*100,2)

    return result,confidence