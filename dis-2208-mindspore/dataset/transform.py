import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor, dtype

# from mindspore.dataset.transforms.py_transforms import Compose
# from mindspore.dataset.vision.c_transforms import Normalize
# from mindspore.dataset.vision.py_transforms import ToTensor

from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import Normalize
from mindspore.dataset.vision import ToTensor


def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    # img = Compose([
    #     ToTensor(),
    #     Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], is_hwc=False),
    # ])(img)

    img_np = np.array(img)
    img = Tensor(img_np)
    img = mnp.true_divide(img, 255.0)

    img_mean = Tensor([0.5, 0.5, 0.5])
    img_std = Tensor([1.0, 1.0, 1.0])
    img = mnp.subtract(img, img_mean)
    img = mnp.divide(img, img_std)
    # img = Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], is_hwc=False)
    img = mnp.transpose(img, (2, 0, 1))
    
    if mask:
        mask_np = np.array(mask)
        mask_tensor = Tensor(mask_np)
        mask = mnp.divide(mask_tensor, 255.0)
        return img, mask
    return img


def resize(img, mask, size):
    img = img.resize((size, size), Image.BILINEAR)
    mask = mask.resize((size, size), Image.BILINEAR)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(
                value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask
