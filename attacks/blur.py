import kornia
from attacks import BaseAttack


class GaussianBlurAttacker(BaseAttack):
    def __init__(self):
        super(GaussianBlurAttacker, self).__init__()
        self.name = 'Gaussian Blur'

    def get_intensity_name(self, intensity=None):
        return f'K.S. = {intensity}'

    def attack_image(self, image, intensity):
        return kornia.filters.gaussian_blur2d(image, kernel_size=(intensity, intensity), sigma=(1, 1))

class MedianBlurAttacker(BaseAttack):
    def __init__(self):
        super(MedianBlurAttacker, self).__init__()
        self.name = 'Median Blur'

    def get_intensity_name(self, intensity=None):
        return f'K.S. = {intensity}'

    def attack_image(self, image, intensity):
        return kornia.filters.median_blur(image, kernel_size=(intensity, intensity))

class AverageBlurAttacker(BaseAttack):
    def __init__(self):
        super(AverageBlurAttacker, self).__init__()
        self.name = 'Average Blur'

    def get_intensity_name(self, intensity=None):
        return f'K.S. = {intensity}'

    def attack_image(self, image, intensity):
        return kornia.filters.box_blur(image, kernel_size=(intensity, intensity))