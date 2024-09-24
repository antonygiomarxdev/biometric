from abc import ABC, abstractmethod


class MinutiaeExtractorRepository(ABC):
    @abstractmethod
    def enhance(self, img):
        pass

    @abstractmethod
    def binarize(self, img):
        pass

    @abstractmethod
    def erode(self, img):
        pass

    @abstractmethod
    def skeletonize(self, img):
        pass

    @abstractmethod
    def extract_minutiae_features(self, img):
        pass
