from abc import ABC, abstractmethod

class StrategyTI(ABC):

    @abstractmethod
    def gaussian_blur(self, image, sigma):
        pass
    @abstractmethod
    def averge_blur(self, image):
        pass