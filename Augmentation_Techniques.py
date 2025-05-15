import numpy as np
from imgaug import augmenters as iaa
import cv2
import matplotlib.image as mpimg

class ImageAugmentor:
    def zoom(self, image):
        zoom = iaa.Affine(scale=(1, 1.3))
        return zoom.augment_image(image)

    def pan(self, image):
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        return pan.augment_image(image)

    def random_brightness(self, image):
        brightness = iaa.Multiply((0.2, 1.2))
        return brightness.augment_image(image)

    def random_flip(self, image, steering_angle):
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
        return image, steering_angle

    def random_augment(self, image_path, steering_angle):
        image = mpimg.imread(image_path)

        if np.random.rand() < 0.5:
            image = self.pan(image)
        if np.random.rand() < 0.5:
            image = self.zoom(image)
        if np.random.rand() < 0.5:
            image = self.random_brightness(image)
        if np.random.rand() < 0.5:
            image, steering_angle = self.random_flip(image, steering_angle)

        return image, steering_angle

    def preprocess(self, img):
        img = img[60:135, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img / 255.0
        return img
