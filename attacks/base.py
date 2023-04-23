class BaseAttack():
    def __init__(self, val_MIN=-1, val_MAX=1):
        self.name = 'wo Attack'
        self.val_MIN = val_MIN
        self.val_MAX = val_MAX

    def attack(self, image, intensity):
        attacked_image = image
        attacked_image = self.normalize(attacked_image)
        attacked_image = self.attack_image(attacked_image, intensity)
        attacked_image = self.denormalize(self.quantize_pipeline(attacked_image))
        return attacked_image

    def attack_image(self, image, intensity):
        return image

    def normalize(self, image):
        image = (image - self.val_MIN) / (self.val_MAX - self.val_MIN)
        return image

    def denormalize(self, image):
        image = image * (self.val_MAX - self.val_MIN) + self.val_MIN
        return image

    def quantize_pipeline(self, image):
        image = image.clamp(0, 1)
        image = (image * 255).int()
        image = image.float() / 255
        return image

    def get_attack_name(self):
        return self.name

    def get_intensity_name(self, intensity=None):
        return ''
