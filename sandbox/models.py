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