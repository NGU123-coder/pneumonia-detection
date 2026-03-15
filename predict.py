import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from lung_segmentation import segment_lungs

# rebuild the model architecture
base_model = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# load trained weights
model.load_weights("model/pneumonia_resnet50.h5")


def predict_pneumonia(img_path):

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