import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import Sequence
import gradcam
import cv2
from tqdm.notebook import tqdm
import albumentations


class PANDAGenerator(Sequence):
    def __init__(self, df, config, mode='fit', apply_tfms=True, shuffle=True):
        super(PANDAGenerator, self).__init__()
        
        self.image_ids = df['image_id'].reset_index(drop=True).values
        self.labels = df['isup_grade'].reset_index(drop=True).values
        
        self.config = config
        self.shuffle = shuffle
        self.mode = mode
        
        self.apply_tfms = apply_tfms
        
        self.side = int(self.config.num_tiles ** 0.5)
        self.input_size= (self.config.img_size*self.side,self.config.img_size*self.side,3)
        self.tfms = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=.1, scale_limit=.1, rotate_limit=20, p=0.5),
        ])
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.config.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_ids))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.zeros((self.config.batch_size, self.side * self.config.img_size, \
                      self.side * self.config.img_size, 3), dtype=np.float32)
        
        imgs_batch = self.image_ids[index * self.config.batch_size : (index + 1) * self.config.batch_size]
        
        for i, img_name in enumerate(imgs_batch):
            try:
                img_path = '{}/{}.png'.format(self.config.backbone_train_path, img_name)
                image= tf.keras.preprocessing.image.load_img(path=img_path,grayscale=False,color_mode="rgb",target_size=(self.config.img_size*self.side,self.config.img_size*self.side),interpolation='nearest')
                input=tf.keras.preprocessing.image.img_to_array(image)
                if(input.shape!=self.input_size):
                    input=cv2.resize(input,self.input_size,interpolation=cv2.INTER_LANCZOS4)
                X[i, ]=input/255.0
            except:
                print(img_name)
        if self.mode == 'fit':
            y = np.zeros((self.config.batch_size, self.config.num_classes), dtype=np.float32)
            lbls_batch = self.labels[index * self.config.batch_size : (index + 1) * self.config.batch_size]
            
            for i in range(self.config.batch_size):
                y[i, lbls_batch[i]] = 1
            return X, y
        
        elif self.mode == 'predict':
            return X
        
        else:
            raise AttributeError('mode parameter error')

    # def __getitem__(self, index):
    #     X = np.zeros((self.config.batch_size, self.side * self.config.img_size, \
    #                   self.side * self.config.img_size, 3), dtype=np.float32)
        
    #     imgs_batch = self.image_ids[index * self.config.batch_size : (index + 1) * self.config.batch_size]
        
    #     for i, img_name in enumerate(imgs_batch):
    #         img_path = '{}/{}.tiff'.format(self.config.backbone_train_path, img_name)
    #         img_patches = self.get_patches(img_path)
    #         X[i, ] = self.glue_to_one(img_patches)
            
    #     if self.mode == 'fit':
    #         y = np.zeros((self.config.batch_size, self.config.num_classes), dtype=np.float32)
    #         lbls_batch = self.labels[index * self.config.batch_size : (index + 1) * self.config.batch_size]
            
    #         for i in range(self.config.batch_size):
    #             y[i, lbls_batch[i]] = 1
    #         return X, y
        
    #     elif self.mode == 'predict':
    #         return X
        
    #     else:
    #         raise AttributeError('mode parameter error')

    # def get_patches(self, img_path):
    #     num_patches = self.config.num_tiles
    #     p_size = self.config.img_size
    #     img = skimage.io.MultiImage(img_path)[-1] / 255
        
    #     if self.apply_tfms:
    #         img = self.tfms(image=img)['image'] 
        
    #     pad0, pad1 = (p_size - img.shape[0] % p_size) % p_size, (p_size - img.shape[1] % p_size) % p_size
        
    #     img = np.pad(
    #         img,
    #         [
    #             [pad0 // 2, pad0 - pad0 // 2], 
    #             [pad1 // 2, pad1 - pad1 // 2], 
    #             [0, 0]
    #         ],
    #         constant_values=1
    #     )
        
    #     img = img.reshape(img.shape[0] // p_size, p_size, img.shape[1] // p_size, p_size, 3)
    #     img = img.transpose(0, 2, 1, 3, 4).reshape(-1, p_size, p_size, 3)
        
    #     if len(img) < num_patches:
    #         img = np.pad(
    #             img, 
    #             [
    #                 [0, num_patches - len(img)],
    #                 [0, 0],
    #                 [0, 0],
    #                 [0, 0]
    #             ],
    #             constant_values=1
    #         )
            
    #     idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_patches]
    #     return np.array(img[idxs])
    
