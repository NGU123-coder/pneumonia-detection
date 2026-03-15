import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from lung_segmentation import segment_lungs

model = load_model("model/pneumonia_resnet50.h5")

last_conv_layer_name = "conv5_block3_out"

def generate_gradcam(img_path,output_path):

    segmented = segment_lungs(img_path)

    img = cv2.resize(segmented,(224,224))

    img_array = img/255.0

    img_array = np.expand_dims(img_array,axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output,model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs,predictions = grad_model(img_array)

        loss = predictions[:,0]

    grads = tape.gradient(loss,conv_outputs)

    pooled_grads = tf.reduce_mean(grads,axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[...,tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap,0) / tf.math.reduce_max(heatmap)

    heatmap = heatmap.numpy()

    original = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap,(original.shape[1],original.shape[0]))

    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original,0.6,heatmap,0.4,0)

    cv2.imwrite(output_path,overlay)

    return output_path