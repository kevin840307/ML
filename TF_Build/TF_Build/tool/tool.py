import numpy as np
from PIL import Image

def array_img_save(array, save_path, normal=True, binary=False):
    if binary:
        array[array > 0.5] = 1
        array[array <= 0.5] = 0

    if normal:
        array = array * 255

    if np.shape(array)[2] == 1:
        array = np.tile(array, [1, 1, 3])

    array = array.astype(np.uint8)
    img = Image.fromarray(array)
    img.save(save_path)