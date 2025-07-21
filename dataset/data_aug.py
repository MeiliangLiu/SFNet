import numpy as np
import random
import scipy.ndimage as ndimage

def random_zoom(img_numpy, min_percentage=0.8, max_percentage=1.1):
    """
    :param img_numpy:
    :param min_percentage:
    :param max_percentage:
    :return: zoom in/out aigmented img
    """
    z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0,0],
                            [0, z, 0,0],
                            [0, 0, z,0],
                            [0, 0, 0,1]])
    return ndimage.interpolation.affine_transform(img_numpy, zoom_matrix)


class RandomZoom(object):
    def __init__(self, min_percentage=0.8, max_percentage=1.1):
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        img_numpy = random_zoom(img_numpy, self.min_percentage, self.max_percentage)
        return img_numpy

def random_noise(img_numpy, mean=0, std=0.001):
    noise = np.random.normal(mean, std, img_numpy.shape)

    return img_numpy + noise


class GaussianNoise(object):
    def __init__(self, mean=0, std=0.001):
        self.mean = mean
        self.std = std

    def __call__(self, img_numpy, label=None):

        return random_noise(img_numpy, self.mean, self.std)
def transform_matrix_offset_center_3d(matrix, x, y, z):
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


def random_shift(img_numpy, max_percentage=0.2):
    dim1, dim2, dim3 = img_numpy.shape
    m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
    d1 = np.random.randint(-m1, m1)
    d2 = np.random.randint(-m2, m2)
    d3 = np.random.randint(-m3, m3)
    return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)


class RandomShift(object):
    def __init__(self, max_percentage=0.2):
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        img_numpy = random_shift(img_numpy, self.max_percentage)

        return img_numpy


class RandomChoice(object):
    """
    choose a random tranform from list an apply
    transforms: tranforms to apply
    p: probability
    """

    def __init__(self, transforms=[],
                 p=0.5):
        self.transforms = transforms
        self.p = p
    # 随机数小于p时，执行变换
    def __call__(self, img_tensors):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors

        t = random.choice(self.transforms)

        img_tensors = t(img_tensors)
        return img_tensors


