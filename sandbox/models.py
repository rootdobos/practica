from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import gradcam
from classification_models.tfkeras import Classifiers
import tensorflow_addons as tfa
import tensorflow as tf
def factory(config):
    if(config.model_name=="seresnext50"):
        return build_seresnext(config)
    if(config.model_name=="vgg16"):
        return build_vgg16(config)
    if(config.model_name=="densenet121"):
        return build_densenet121(config)
    if(config.model_name=="efficientNetB2"):
        return build_efficientNetB2(config)
    
def build_seresnext(config):
        
    SEResNEXT50, _ = Classifiers.get('seresnext50')
    base_model = SEResNEXT50(input_shape=(config.img_size*int(config.num_tiles**0.5), \
                                          config.img_size*int(config.num_tiles**0.5), 3), \
                             weights='imagenet', include_top=False)
    
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(config.num_classes, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config.learning_rate), \
              metrics=[tfa.metrics.CohenKappa(weightage='quadratic', num_classes=6)])
    
    return model
def build_densenet121(config):
        
    base_model = tf.keras.applications.DenseNet121(input_shape=(config.img_size*int(config.num_tiles**0.5), \
                                          config.img_size*int(config.num_tiles**0.5), 3), \
                             weights='imagenet', include_top=False)
    
    x = GlobalAveragePooling2D()(base_model.output)
    dropout= Dropout(0.80)(x)
    output = Dense(config.num_classes, activation='softmax')(dropout)
    model = Model(inputs=[base_model.input], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config.learning_rate), \
              metrics=[tfa.metrics.CohenKappa(weightage='quadratic', num_classes=6)])
    
    return model

def build_efficientNetB2(config):
        
    base_model = tf.keras.applications.EfficientNetB2(input_shape=(config.img_size*int(config.num_tiles**0.5), \
                                          config.img_size*int(config.num_tiles**0.5), 3), \
                             weights='imagenet', include_top=False)
    
    x = GlobalAveragePooling2D()(base_model.output)
    dropout= Dropout(0.40)(x)
    output = Dense(config.num_classes, activation='softmax')(dropout)
    model = Model(inputs=[base_model.input], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config.learning_rate), \
              metrics=[tfa.metrics.CohenKappa(weightage='quadratic', num_classes=6)])
    
    return model

def build_vgg16(config):
    base_model = tf.keras.applications.VGG16(input_shape=(config.img_size*int(config.num_tiles**0.5),
                                                          config.img_size*int(config.num_tiles**0.5), 3),
                                             include_top=False,
                                             weights='imagenet')
    
    base_model.trainable = True
    x=GlobalAveragePooling2D()(base_model.output)
    dense1=Dense(16, activation='relu')(x)
    batchnorm= tf.keras.layers.BatchNormalization()(dense1)
    output= Dense(config.num_classes, activation='softmax')(batchnorm)

    # model = tf.keras.Sequential([
    #     base_model,
        
    #     tf.keras.layers.GlobalAveragePooling2D(),
    #     tf.keras.layers.Dense(16, activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(config.num_classes, activation='softmax'),
    # ])
    model = Model(inputs=[base_model.input], outputs=[output])


    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc'),tfa.metrics.CohenKappa(weightage='quadratic', num_classes=6)])
    
    return model