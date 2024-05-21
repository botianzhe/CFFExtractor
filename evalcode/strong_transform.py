from torchvision import transforms
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,JpegCompression,
    Transpose, ShiftScaleRotate, Blur,GaussianBlur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,RandomBrightness,ToSepia
)
import numpy as np

def strong_aug(p=0.5):
    return Compose([
        HorizontalFlip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.25),
            JpegCompression(50,90),
            GaussianBlur(p=0.5),
            Blur(blur_limit=3, p=0.25),
        ], p=0.2),
        HueSaturationValue(p=0.2),
        OneOf([
            RandomBrightness(),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.6),
        ToSepia(p=0.1)
    ], p=p)
augmentation=strong_aug()
trans=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.38348666],std=[0.20834281])
            transforms.Normalize(mean=[0.38348666, 0.39193852, 0.4665315],std=[0.20834281, 0.20540032, 0.24183848])
        ])
