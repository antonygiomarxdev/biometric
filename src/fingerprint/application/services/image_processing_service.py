class ImageProcessingService:
    def __init__(self, extractor_impl):
        self.extractor_impl = extractor_impl

    def enhance_image(self, img):
        return self.extractor_impl.enhance(img)

    def binarize_image(self, img):
        return self.extractor_impl.binarize(img)

    def erode_image(self, img):
        return self.extractor_impl.erode(img)

    def skeletonize_image(self, img):
        return self.extractor_impl.skeletonize(img)
