import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from lung_segmentation import segment_lungs

# rebuild architecture
base_model = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# load trained weights
model.load_weights("model/pneumonia_resnet50.h5")

last_conv_layer_name = "conv5_block3_out"


def generate_gradcam(img_path, output_path):

    segmented = segment_lungs(img_path)

    img = cv2.resize(segmented,(224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:,0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap,0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    original = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap,(original.shape[1], original.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original,0.6,heatmap,0.4,0)

    cv2.imwrite(output_path,overlay)

    return output_path