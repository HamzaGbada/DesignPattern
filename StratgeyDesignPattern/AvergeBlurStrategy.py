from StratgeyDesignPattern.Strategy import StrategyTI
import numpy as np
from StratgeyDesignPattern import Convolution


class AvergeBlurStrategy(StrategyTI):

    def averge_blur(self, image, mask_size):
        mask = np.ones((mask_size, mask_size)) * 1.0 / (mask_size**2)
        im_filtered = Convolution.convolution(image, mask, True)
        return (im_filtered.astype(np.uint8))