import numpy as np
import os, gc, time, random
import uuid


import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model
import cv2

from math import ceil
import warnings
warnings.filterwarnings('ignore')




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_axis_max_min(array, axis=0):
    one_axis = list((array != 255).sum(axis=tuple([x for x in (0, 1, 2) if x != axis])))
    axis_min = next((i for i, x in enumerate(one_axis) if x), 0)
    axis_max = len(one_axis) - next((i for i, x in enumerate(one_axis[::-1]) if x), 0)
    return axis_min, axis_max

def get_img_array(img_path, config):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(config.img_size, config.img_size))
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0) # Add one dimension to transform our array into a batch
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)
    
    classifier_input = Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        
        preds = classifier_model(last_conv_layer_output)
        print(preds)
        # top_pred_index = tf.argmax(preds[0])
        # top_class_channel = preds[:, top_pred_index]
        
    grads = tape.gradient(preds, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, preds

def create_superimposed_visualization(img, heatmap, colormap, label,prediction ,alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap *=(255.0/heatmap.max())
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap,colormap )

    image =img *(255.0/img.max())
    superimposed_img = np.uint8( heatmap *alpha + (1-alpha)*image)
    
    guid=str( uuid.uuid4())


    cv2.imwrite("gradcam_images/"+guid+"_label_"+str( label)+".png",cv2.cvtColor( np.uint8(image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("gradcam_images/"+guid+"_pred_"+str(np.uint8(prediction))+".png", cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_RGB2BGR))

    return superimposed_img
    return heatmap
def make_gradcam_heatmap_orig(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index

