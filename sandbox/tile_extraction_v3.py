import os
import numpy as np
import pandas as pd
import warnings
import math
from multiprocessing import Pool
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from openslide import OpenSlide
import cv2
import utils3
import random
from openslide import open_slide
import openslide
from openslide.deepzoom import DeepZoomGenerator
import uuid
IMG_DIR = 'E:/data/prostate_cancer/train_images/'
TILES_BASE_DIR="E:/data/prostate_cancer/tiles/"
# IMG_DIR = 'train_images/'
# TILES_BASE_DIR="tiles/"
class TileExtractor:
    def __init__(self, layer, size, cols,rows,augmentation):
        self.layer = layer
        self.size = size
        self.cols=cols
        self.rows=rows
        self.n=cols*rows
        self.augmentation=augmentation
        self.tiles_dir="{}{}_{}_{}_{}_{}".format(TILES_BASE_DIR,self.layer,self.size,self.cols,self.rows,self.augmentation)

    def process_image(self,idx):    
        if os.path.exists(os.path.join(self.tiles_dir, f"{idx}.png")):
            return
        #im = utils3.imread(os.path.join(IMG_DIR, f"{idx}.tiff"), layer=self.layer)
        slide=open_slide( os.path.join(IMG_DIR, f"{idx}.tiff")) 
        tiles = self.akensert_tiles(slide)
        im = self.join_tiles(tiles)
        im = Image.fromarray(im)
        im.save(os.path.join(self.tiles_dir, f"{idx}.png"), format='PNG', quality=90)
        
        for i in range(self.augmentation):
            np.random.shuffle(tiles)
            for j in range(self.n):
                rand=random.randrange(0,4)
                if(rand==1):
                    tiles[j]= cv2.rotate(tiles[j],cv2.ROTATE_90_CLOCKWISE)
                if(rand==2):
                    tiles[j]= cv2.rotate(tiles[j],cv2.ROTATE_90_COUNTERCLOCKWISE)
                if(rand==3):
                    tiles[j]= cv2.rotate(tiles[j],cv2.ROTATE_180)
            im = self.join_tiles(tiles)
            im = Image.fromarray(im)
            im.save(os.path.join(self.tiles_dir, f"{idx}_{i}.png"), format='PNG', quality=90)

    # def split_tiles(img:np.ndarray)->np.ndarray:
    #     reshaped = img.reshape(
    #         img.shape[0] // SIZE,
    #         SIZE,
    #         img.shape[1] // SIZE,
    #         SIZE,
    #         3,
    #     )
    #     transposed = reshaped.transpose(0, 2, 1, 3, 4)
    #     return transposed.reshape(-1, SIZE, SIZE, 3)

    def join_tiles(self,img:np.ndarray)->np.ndarray:
        reshaped = img.reshape(
            self.cols,
            self.rows,    
            img.shape[1],
            img.shape[2],
            3
        )
        transposed = reshaped.transpose(0, 2, 1, 3, 4)
        return transposed.reshape(self.cols * self.size, self.rows * self.size, 3)

    # def lafoss_tiles(img:np.ndarray)->np.ndarray:
        
    #     # calculate paddings
    #     H, W, _ = img.shape
    #     pad_w = (SIZE - W % SIZE) % SIZE
    #     pad_h = (SIZE - H % SIZE) % SIZE
        
    #     # implement padding
    #     padded = np.pad(
    #         img,
    #         [[pad_h // 2, pad_h - pad_h // 2],
    #         [pad_w // 2, pad_w - pad_w // 2],
    #         [0, 0]],
    #         constant_values=255, # 255 - white
    #     )
        
    #     # split image into tiles
    #     tiles = split_tiles(padded)
        
    #     # calculate sums of all pixsels for each tile
    #     sums = tiles.reshape(tiles.shape[0], -1).sum(axis=-1)
        
    #     # take top N tiles by minimum sum value
    #     idxs_selected = np.argsort(sums)[:N]
    #     selected = tiles[idxs_selected]
        
    #     # append white tiles if necessary
    #     if len(selected)<N:
    #         selected = np.pad(
    #             selected,
    #             [[0,N-len(selected)],[0,0],[0,0],[0,0]],
    #             constant_values=255
    #         )
        
    #     # join selected tiles into one image
    #     merged = join_tiles(selected)

    #     return merged

    def akensert_tiles(self,slide, debug=False)->np.ndarray:    
        # get tile coords
        downsampled_image=slide.read_region((0,0),2, slide.level_dimensions[2])
        downsampled_image= downsampled_image.convert('RGB')
        downsampled_image = np.array(downsampled_image)
        scaled_patch_size=int( self.size/slide.level_downsamples[2])
        coords = utils3.compute_coords(
            downsampled_image,
            patch_size=scaled_patch_size,
            precompute=True, # returns new padded img
            min_patch_info=0.35,
            min_axis_info=0.35,
            min_consec_axis_info=0.35,
            min_decimal_keep=0.7)
        # sort coords (high info -> low info)
        coords = sorted(coords, key= lambda x: x[0], reverse=False)
        zoom_tiles= DeepZoomGenerator(slide,tile_size=self.size,overlap=0,limit_bounds=False)
        # select top N tiles
        level_num = zoom_tiles.level_count-1
        
        tiles = []
        for i in range(len(coords)):
            if i == self.n:
                break;
            _, x, y = coords[i]
            scaled_row=int(x*slide.level_downsamples[2]/self.size)
            scaled_col=int(y*slide.level_downsamples[2]/self.size)

            temp_tile=zoom_tiles.get_tile(level_num,(scaled_col,scaled_row))

            temp_tile_RGB = temp_tile.convert('RGB')
            temp_tile_np = np.array(temp_tile_RGB)
            correction_x=self.size-temp_tile_np.shape[0]
            correction_y=self.size-temp_tile_np.shape[1]
            if(correction_y>0 or correction_x>0):
                # id=uuid.uuid4().hex
                # im = Image.fromarray(temp_tile_np)
                # im.save(os.path.join("C:/tmp/tiles_bug", f"{id}_orig.png"), format='PNG', quality=90)
                temp_tile_np=np.pad(temp_tile_np,((0,correction_x),(0,correction_y), (0,0)),'maximum')
                # im = Image.fromarray(temp_tile_np)
                # im.save(os.path.join("C:/tmp/tiles_bug", f"{id}_padded.png"), format='PNG', quality=90)
            tiles.append(temp_tile_np)
        
        # append white tiles if necessary
        selected = np.array(tiles)
        if len(selected)<self.n:
            selected = np.pad(
                selected,
                [[0,self.n-len(selected)],[0,0],[0,0],[0,0]],
                constant_values=255
            )
        return selected
        # merge tiles to one image
        merged = join_tiles(selected)
        
        if debug:
            for (v, y, x) in coords:
                img = cv2.rectangle(img, (x, y), (x+SIZE, y+SIZE), color=(0, 0, 0), thickness=5)
                img = cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
                img = cv2.circle(img, (x+SIZE, y+SIZE), radius=5, color=(0, 255, 0), thickness=-1)
            return merged, img
        else:
            return merged