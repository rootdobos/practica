import numpy as np
import os, gc, time, random

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model
import cv2

from math import ceil
import warnings
warnings.filterwarnings('ignore')


class config:
    def __init__(self, 
            model_name,
            seed ,
            batch_size,
            img_size ,
            num_tiles,
            num_classes,
            num_splits ,
            num_epochs ,
            learning_rate,
            num_workers ,
            verbose ,
            train_data_name,
            last_conv_layer_name,
            classifier_layer_names):
        self.train_data_name=train_data_name
        self.backbone_train_path='tiles/{}/'.format(train_data_name)
        self.model_name=model_name
        self.seed = seed
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_tiles = num_tiles
        self.num_classes = num_classes
        self.num_splits = num_splits
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.verbose = verbose
        self.train_csv = "tiles/{}.csv".format(train_data_name)
        self.last_conv_layer_name=last_conv_layer_name
        self.classifier_layer_names=classifier_layer_names
        #backbone_test_path = '../input/prostate-cancer-grade-assessment/test_images/'

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
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
        
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, top_pred_index

def create_superimposed_visualization(img, heatmap, colormap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap,colormap )
    superimposed_img = heatmap * 0.4 + img
    
    return superimposed_img