"""
Defines augmentations, train generator and val/test data alignment for model input.

Resources:
  https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/train.py
  https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

Author: Jake Lepere
Date: 08/29/2018
"""

import numpy as np
from random import randint, uniform, shuffle, random
from skimage.util import crop, random_noise
from skimage.transform import resize, rotate, warp, AffineTransform
from skimage.exposure import rescale_intensity
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence


def randomHorizontalFlip(image, mask, u=0.5):
    """
    Randomly flip the image and mask.

    Args:
      image: the image
      mask: the mask
      u: the augmentation probability
    """
    if random() < u:
        image = image[:,::-1,:]
        mask = mask[:,::-1,:]
    return image, mask


def randomVerticalFlip(image, mask, u=0.5):
    """
    Randomly flip the image and mask.

    Args:
      image: the image
      mask: the mask
      u: the augmentation probability
    """
    if random() < u:
        image = image[::-1,:,:]
        mask = mask[::-1,:,:]
    return image, mask


def randomZoom(image, mask, m, u=0.5):
    """
    Randomly zoom the image and mask by trimming the top, bottom, left and right.

    Args:
      image: the iamge
      mask: the mask
      m: the maximum to trim from the top, bottom, left and right
      u: the augmentation probability
    """
    if random() < u:
        original_shape = image.shape
        c = ((randint(0,m),randint(0,m)),(randint(0,m),randint(0,m)),(0,0))
        image = resize(crop(image, crop_width=c), original_shape)
        mask = resize(crop(mask, crop_width=c), original_shape)
    return image, mask


def randomRotate(image, mask, m, mode='constant', u=0.5):
    """
    Randomly rotate the image and mask.

    Args:
      image: the image
      mask: the mask
      m: the max rotation angle [-a,a]
      u: the augmentation probability
    """
    if random() < u:
        r = uniform(-1*m, m)
        image = rotate(image, angle=r, mode=mode)
        mask = rotate(mask, angle=r, mode=mode)
    return image, mask


def randomTranslate(image, mask, m, mode='constant', u=0.5):
    """
    Randomly translate the image and mask, in the X direction only.

    Args:
      image: the image
      mask: the mask
      m: the max translation up, down, left or right
      u: the augmentation probability
    """
    if random() < u:
        t = (uniform(-1*m, m), 0)
        tf = AffineTransform(translation=t)
        image = warp(image, tf, mode=mode)
        mask = warp(mask, tf, mode=mode)
    return image, mask


def randomShear(image, mask, m, mode='constant', u=0.5):
    """
    Randomly shear the image and mask.

    Args:
      image: the image
      mask: the mask
      m: the max shear
      u: the augmentation probability
    """
    if random() < u:
        s = uniform(-1*m, m)
        tf = AffineTransform(shear=s)
        image = warp(image, tf, mode=mode)
        mask = warp(mask, tf, mode=mode)
    return image, mask


def randomNoise(image, m, u=0.5):
    """
    Randomly add noise to the image.

    Args:
      image: the image
      m: tuple representing the random lower and upper bound for the amount of noise
      u: the augmentation probability
    """
    return random_noise(image, mode='s&p', amount=uniform(m[0],m[1])) if random() < u else image


def randomIntensityRescale(image, m, u=0.5):

    if random() < u:
        if random() < 0.5:
            image = rescale_intensity(image, in_range=(uniform(0.,m),1.)) # lighten
        else:
            image = rescale_intensity(image, in_range=(0.,uniform(1.-m,1.))) # darken
    return image


class DataGenerator(Sequence):
    """
    DataGenerator class for the TGS Competition.
    """

    def __init__(self, images, masks, depths, batch_size=32, shuffle=True):
        """
        Initializes the generator.

        Args:
          images: the images
          masks: the image masks
          depths: the image depths
          batch_size: the size of the batch
          shuffle: True to shuffle the generator output, false otherwise
        """

        # initialize
        self.images = images
        self.masks = masks
        self.depths = depths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_examples = images.shape[0]
        self.image_height = images.shape[1]
        self.image_width = images.shape[2]
        self.image_channels = images.shape[3]
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.num_examples / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_examples)
        if self.shuffle == True:
            shuffle(self.indexes)

    def __data_generation(self, indexes):
        num_indexes = len(indexes)
        image_batch = np.zeros((num_indexes, self.image_height, self.image_width, self.image_channels))
        mask_batch = np.zeros((num_indexes, self.image_height, self.image_width, self.image_channels))
        depth_batch = np.zeros((num_indexes, 1))
        row_batch = get_rows((self.image_height, self.image_width, self.image_channels), num_indexes)

        for i,j in enumerate(indexes):
            image = self.images[j]
            mask = self.masks[j]
            depth = self.depths[j]
            image, mask = randomHorizontalFlip(image, mask, u=0.5)
            #image, mask = randomVerticalFlip(image, mask, u=0.5)
            #image = randomIntensityRescale(image, 0.03, u=1.)
            image, mask = randomZoom(image, mask, 2, u=1.)
            image, mask = randomRotate(image, mask, 3, mode='edge', u=1.)
            image, mask = randomTranslate(image, mask, 1, mode='edge', u=1.)
            #image, mask = randomShear(image, mask, 0.05, mode='edge', u=1.)
            #image = randomNoise(image, (0.05, 0.05), u=0.08)
            image_batch[i] = image
            mask_batch[i] = mask
            depth_batch[i] = self.depths[j]
        
        return [image_batch, depth_batch, row_batch], mask_batch


def organize_val_data(images, masks, depths):
    """
    Organizes validation data for appropriate model input.

    Args:
      images: validation images
      masks: validation masks
      depths: validation depths
    """
    return ([images, depths, get_rows(images.shape[1:], images.shape[0])], masks)


def organize_test_data(images, depths):
    """
    Organizes testing data for appropriate model input.

    Args:
      images: testing images
      depths: testing depths
    """
    return [images, depths, get_rows(images.shape[1:], images.shape[0])]


def get_rows(image_shape, batch_size):
    """
    Gets a numpy array representing the row at each pixel, normalized by the height of the image.
    For example:
      [[.0,.0,...,.0,.0]
       [.1,.1,...,.1,.1]
       ...
       [.9,.9,...,.9,.9]
       [1.,1.,...,1.,1.]]

    Args:
      image_shape: the shape of the images to match dimensions
      batch_size: the size of the batch
    """
    h = image_shape[0]
    w = image_shape[1]
    c = image_shape[2]
    b = batch_size
    return np.array([[r/h for r in range(1,h+1) for _ in range(w)] for _ in range(b)]).reshape((b, h, w, c))

