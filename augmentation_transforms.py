
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image

IMAGE_SIZE = 32
MEANS = [0.49139968, 0.48215841, 0.44653091]
STDS = [0.24703223, 0.24348513, 0.26158784]
PARAMETER_MAX = 10   # What is the max 'level' a transform could be predicted


def random_flip(x):
    if np.random.rand(1)[0] > 0.5:
        return np.fliplr(x)
    return x

def zero_pad_and_crop(img, amount=4):
    padded_img = np.zeros((img.shape[0] + amount * 2, img.shape[1] + amount * 2, img.shape[2]))
    padded_img[amount:img.shape[0] + amount, amount:img.shape[1] + amount, :] = img
    top = np.random.randint(low=0, high = 2 * amount)
    left = np.random.randint(low=0, high =2 * amount)
    new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
    return new_img
    
def create_cutout_mask(img_height, img_width, num_channels, size):
    assert img_height == img_width
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)
    # print('height_loc', height_loc,'  width_loc:  ', width_loc, 'size:        ',size)
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    # print('height_loc', upper_coord)
    lower_coord = (min(img_height, height_loc + size // 2), min(img_width, width_loc + size // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0
    
    mask = np.ones((img_height, img_width, num_channels))
    zeros = np.zeros((mask_height, mask_width, num_channels))
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (zeros)
    return mask, upper_coord, lower_coord

def cutout_numpy(img, size=16):
    img_height, img_width, num_channels = (img.shape[0], img.shape[1], img.shape[2])
    assert len(img.shape) == 3
    mask, _, _ = create_cutout_mask(img_height, img_width, num_channels, size)
    return img * mask

def float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX

def int_parameter(level, maxval):
    return  int(level * maxval / PARAMETER_MAX )

def pil_wrap(img):
    return Image.fromarray(np.uint8((img * STDS + MEANS) * 255.0)).convert('RGBA')

def pil_unwrap(pil_img):
    pic_array = (np.array(pil_img.getdata()).reshape((32, 32, 4)) / 255.0)
    i1, i2 = np.where(pic_array[:, :, 3] == 0)
    pic_array =(pic_array[:, :, :3] - MEANS) / STDS
    pic_array[i1, i2] = [0, 0, 0]
    return pic_array


def apply_policy(policy, img):
    pil_img = pil_wrap(img)
    for xform in policy:
        assert len(xform) == 3
        name, probability, level = xform
        xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level)
        pil_img = xform_fn(pil_img)
    return pil_unwrap(pil_img)


class TransformFunction(object):
    def __init__(self, func, name):
        self.f = func
        self.name = name
        
    def __repr__(self):
        return '<' + self.name +'>'
    
    def __call__(self, pil_img):
        return self.f(pil_img)

    
class TransformT(object):
    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn
        
    def pil_transformer(self, probability, level):
        def return_function(im):
            if random.random() < probability:
                im = self.xform(im, level)
            return im
        
        name = self.name + '({:.1f}, {})'.format(probability, level)
        return TransformFunction(return_function, name)
    
    def do_transform(self, image, level):
        f = self.pil_transformer(PARAMETER_MAX, lever)
        return pil_unwrap(f(pil_wrap(image)))
 



 ################## Transform Functions ##################
identity = TransformT('identity', 
                      lambda pil_img, level: pil_img)

flip_lr = TransformT('FlipLR',
                     lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))

flip_ud = TransformT('FlipUD',
                     lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))

# pylint:disable=g-long-lambda
# Maximize the the image contrast, by making the darkest pixel black and lightest pixel white.
auto_contrast = TransformT('AutoContrast',
                           lambda pil_img, level: ImageOps.autocontrast(pil_img.convert('RGB')).convert('RGBA'))

# Equalize the image histogram.
equalize = TransformT('Equalize',
                          lambda pil_img, level: ImageOps.equalize(pil_img.convert('RGB')).convert('RGBA'))

# Invert the pixels of the image.
invert = TransformT('Invert',
                    lambda pil_img, level: ImageOps.invert(pil_img.convert('RGB')).convert('RGBA'))

# pylint:enable=g-long-lambda
blur = TransformT('Blur',
                  lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))

smooth = TransformT('Smooth',
                    lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))


def _rotate_impl(pil_img, level):
    degrees = int_parameter(level, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


# Rotate the image magnitude degrees.  Range of magnitudes:  [-30,30]
rotate = TransformT('Rotate', _rotate_impl)


def _posterize_impl(pil_img, level):
    level = int_parameter(level, 4)
    return ImageOps.posterize(pil_img.convert('RGB'), 4 - level).convert('RGBA')
posterize = TransformT('Posterize', _posterize_impl)


def _shear_x_impl(pil_img, level):
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE,  (1, level, 0, 0, 1, 0))

# Shear the image along the horizontal (vertical) axis with rate magnitude. Range of magnitudes:  [-0.3,0.3]
shear_x = TransformT('ShearX', _shear_x_impl)


def _shear_y_impl(pil_img, level):
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, level, 1, 0))
shear_y = TransformT('ShearY', _shear_y_impl)


def _translate_x_impl(pil_img, level):
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, 0, level, 0, 1, 0))

# Translate the image in the horizontal (vertical) direction by magnitude number of pixels. Range of magnitudes: [-150,150]
translate_x = TransformT('TranslateX', _translate_x_impl)

def _translate_y_impl(pil_img, level):
    level = int_parameter(level, 10)
    if random.random() > 0.5:
         level = - level
    return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, 0, 1, level))
translate_y = TransformT('TranslateY', _translate_y_impl)

def _crop_impl(pil_img, level, interpolation=Image.BILINEAR):
    cropped = pil_img.crop((level, level, IMAGE_SIZE - level, IMAGE_SIZE - level))
    resized = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), interpolation)
    return resized
crop_bilinear = TransformT('CropBilinear', _crop_impl)


def _solarize_impl(pil_img, level):
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img.convert('RGB'), 256 - level).convert('RGBA')

# Invert all pixels above a threshold value of magnitude. Range of magnitudes: [0,256]
solarize = TransformT('Solarize', _solarize_impl)

def _cutout_pil_impl(pil_img, level):
    size = int_parameter(level, 20)
    if size == 0:
        return pil_img
    img_height, img_width, num_channels = (32, 32, 3)
    _, upper_coord, lower_coord = (create_cutout_mask(img_height, img_width, num_channels, size))
    pixels = pil_img.load()
    for i in range(upper_coord[0], lower_coord[0]):
        for j in range(upper_coord[1], lower_coord[1]):
            pixels[i, j] = (125, 122, 113, 0)
    return pil_img

# Set a random square patch of side-length magnitude pixels to gray. Range of magnitudes: [0,60]
cutout = TransformT('Cutout', _cutout_pil_impl)


def _enhancer_impl(enhancer):
    def impl(pil_img, level):
        v = float_parameter(level, 1.8) + .1
        return enhancer(pil_img).enhance(v)
    return impl

# Adjust the color balance of the image, in a manner similar to the controls on a colour TV set. 
# A magnitude=0 gives a black & white image, whereas magnitude=1 gives the original image.
# Range of magnitudes: [0.1,1.9]
color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))

# Control the contrast of the image. A magnitude=0 gives a gray image, whereas magnitude=1 gives the original image.
# Range of magnitudes:  [0.1,1.9]
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))

# Adjust the brightness of the image. A magnitude=0 gives a black image, whereas magnitude=1 gives the original image.
# Range of magnitudes:  [0.1,1.9]
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
 
# Adjust the sharpness of the image. A magnitude=0 gives a blurred image, whereas magnitude=1 gives the original image.
# Range of magnitudes:  [0.1,1.9]
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

ALL_TRANSFORMS = [
    flip_lr,
    flip_ud,
    auto_contrast,
    equalize,
    invert,
    rotate,
    posterize,
    crop_bilinear,
    solarize,
    color,
    contrast,
    brightness,
    sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    cutout,
    blur,
    smooth
]

NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}
TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
