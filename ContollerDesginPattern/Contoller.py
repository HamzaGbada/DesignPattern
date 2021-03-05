from StratgeyDesignPattern.AvergeBlurStrategy import AvergeBlurStrategy
from StratgeyDesignPattern.GaussianBlurStrategy import GaussianBlurStrategy
from AdapterDesignPattern.Adapter import Adapter

class Contoller:
    def process(self, image):
        gray = Adapter.rgb_to_gray(image)
        normalized = Adapter.normalization(gray)
        averge_blur = AvergeBlurStrategy.averge_blur(normalized,3)
        gaussian_blur = GaussianBlurStrategy.gaussian_blur(averge_blur, 0.5)
        return gaussian_blur