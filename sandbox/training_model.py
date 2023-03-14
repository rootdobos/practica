import gradcam
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import pandaGenerator
import models

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
def train_model(config,model):
    

    gradcam.seed_everything(config.seed)

    df = pd.read_csv(config.train_csv)
    df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=config.seed)

    train_datagen = pandaGenerator.PANDAGenerator(
        df=train_df, 
        config=config,
        mode='fit', 
        apply_tfms=False,
        shuffle=True, 
    )

    val_datagen = pandaGenerator.PANDAGenerator(
        df=valid_df, 
        config=config,
        mode='fit', 
        apply_tfms=False,
        shuffle=False, 
    )



    cb1 = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, verbose=1, min_lr=1e-6)
    cb2 = ModelCheckpoint("modelcheckpoints/{}/{}.h5".format(config.train_data_name,config.model_name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

    history = model.fit_generator(
        train_datagen,
        validation_data=val_datagen,
        callbacks=[cb1, cb2],
        epochs=config.num_epochs,
        verbose=1
    )
    return history


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn