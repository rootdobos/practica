import numpy as np
import pandas as pd
import math
def pad_image(image, pad_len, pad_val):
    """Pads inputted image, accepts both 
    2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len)), pad_val)
    elif image.ndim == 3:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), pad_val)
    return None

def transpose_image(image):
    """Inputs an image and transposes it, accepts 
    both 2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.transpose(image, (1, 0)).copy()
    elif image.ndim == 3:
        return np.transpose(image, (1, 0, 2)).copy()
    return None
def get_tissue_parts_indices(tissue, min_consec_info):
    """If there are multiple tissue parts in 'tissue', 'tissue' will be 
    split. Each tissue part will be taken care of separately (later on), 
    and if the tissue part is less than min_consec_info, it's considered 
    to small and won't be returned.
    """
    split_points = np.where(np.diff(tissue) != 1)[0]+1
    tissue_parts = np.split(tissue, split_points)
    return [
        tp for tp in tissue_parts if len(tp) >= min_consec_info
    ]
def get_tissue_subparts_coords(subtissue, patch_size, min_decimal_keep):
    """Inputs a tissue part resulting from '_get_tissue_parts_indices'.
    This tissue part is divided into N subparts and returned.
    Argument min_decimal_keep basically decides if we should reduce the
    N subparts to N-1 subparts, due to overflow.
    """
    start, end = subtissue[0], subtissue[-1]
    num_subparts = (end-start)/patch_size
    if num_subparts % 1 < min_decimal_keep and num_subparts >= 1:
        num_subparts = math.floor(num_subparts)
    else:
        num_subparts = math.ceil(num_subparts)

    excess = (num_subparts*patch_size) - (end-start)
    shift = excess // 2

    return [
        i * patch_size + start - shift 
        for i in range(num_subparts)
    ]

def eval_and_append_xy_coords(coords,
                               image, 
                               mask, 
                               patch_size, 
                               x, y, 
                               min_patch_info,
                               transposed,
                               precompute):
    """Based on computed x and y coordinates of patch: 
    slices out patch from original image, flattens it,
    preprocesses it, and finally evaluates its mask.
    If patch contains more info than min_patch_info,
    the patch coordinates are kept, along with a value 
    'val1' that estimates how much information there 
    is in the patch. Smaller 'val1' assumes more info.
    """
    patch_1d = (
        image[y: y+patch_size, x:x+patch_size, :]
        .mean(axis=2)
        .reshape(-1)
    )
    idx_tissue = np.where(patch_1d <= 210)[0]
    idx_black = np.where(patch_1d < 5)[0]
    idx_background = np.where(patch_1d > 210)[0]

    if len(idx_tissue) > 0:
        patch_1d[idx_black] = 210
        patch_1d[idx_background] = 210
        val1 = int(patch_1d.mean())
        val2 = mask[y:y+patch_size, x:x+patch_size].mean()
        if val2 > min_patch_info:
            if precompute:
                if transposed:
                    coords = np.concatenate([
                        coords, [[val1, x-patch_size, y-patch_size]]
                    ])
                else:
                    coords = np.concatenate([
                        coords, [[val1, y-patch_size, x-patch_size]]
                    ])
            else:
                coords = np.concatenate([
                    coords, [[val1, y, x]]
                ])
               
    return coords

