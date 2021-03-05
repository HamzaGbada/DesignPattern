from StratgeyDesignPattern.Strategy import StrategyTI
import numpy as np
from StratgeyDesignPattern import Convolution


class GaussianBlurStrategy(StrategyTI):

    def gaussian_blur(self, image, sigma):
        filter_size = 2 * int(4 * sigma + 0.5) + 1
        gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
        m = filter_size // 2
        n = filter_size // 2
        for x in range(-m, m + 1):
            for y in range(-n, n + 1):
                x1 = 2 * np.pi * (sigma ** 2)
                x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                gaussian_filter[x + m, y + n] = (1 / x1) * x2

        im_filtered = np.zeros(image.shape, dtype=np.float32)
        im_filtered[:, :] = Convolution.convolution(image[:, :], gaussian_filter)
        return (im_filtered.astype(np.uint8))