import torchvision.transforms as T
import cv2
import numpy as np


class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    if is_train:
        color_transform = T.Compose([get_color_distortion(), RandomGaussianBlur()])
        transform = T.Compose(
            [
                T.Resize(size=cfg.INPUT.ORIGIN_SIZE[0]),
                T.RandomResizedCrop(
                    scale=cfg.INPUT.CROP_SCALE, size=cfg.INPUT.CROP_SIZE[0]
                ),
                T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
                color_transform,
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize(size=cfg.INPUT.ORIGIN_SIZE[0]),
                T.CenterCrop(cfg.INPUT.CROP_SIZE[0]),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform
