import numpy as np
from numpy import linalg as la

class Adapter:
    def rgb_to_gray(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    def normalization(self, image):
        norm = la.norm(image)
        image = image.astype('float32')
        image /= norm
        return image