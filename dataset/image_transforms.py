
from torchvision import transforms as T
import numpy as np
from albumentations import (
    HorizontalFlip, VerticalFlip, Compose, ReplayCompose, RandomCrop, ElasticTransform, Rotate,
    GaussNoise, CenterCrop, Resize, RandomScale, ColorJitter, Blur
)
import torch
import random

class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        xi = get_image(self.transform(x))
        xj = get_image(self.transform(x))

        return {'image': torch.stack((xi, xj))}

def get_image(t):
    return torch.Tensor(t['image'])

def flip_horizontal(inp):
    f = HorizontalFlip(always_apply=True)
    return np.array([f(image=slic)['image'] for slic in inp])

def center_crop(inp, size):
    f = CenterCrop(*size, always_apply=True)
    return np.array([f(image=slic)['image'] for slic in inp])

def com_crop(inp, size):
    f = CenterCrop(*size, always_apply=True)
    return np.array([f(image=slic)['image'] for slic in inp])

def center_crop_and_resize(inputs, crop_size, final_size):
    f = Compose([CenterCrop(*crop_size, always_apply=True), Resize(*final_size, interpolation=1, always_apply=True)])
    return [f(image=input)['image'] for input in inputs]

class RandomApply(torch.nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def contrastive_transforms(params):
    f = torch.nn.Sequential(
            RandomApply(T.ColorJitter(params['brightness'], params['contrast'], params['saturation'], params['hue']), p = 0.8),
            # T.RandomGrayscale(p=0.2),
            T.RandomRotation(params['rotate']),
            T.CenterCrop((np.array(params['post_rotate_size']) * params['image_scale']).tolist()),
            T.RandomHorizontalFlip(),
            # RandomApply(
            #     T.GaussianBlur((3, 3), (0.1, 2.0)),
            #     p = 0.2
            # ),
            T.RandomResizedCrop((np.array(params['data_aug_shape']) * params['image_scale']).tolist(), scale=(params['min_crop_area'], 1.0)),
            # T.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225])),
        )
    
    data_augment = lambda x: {'image': f(x)}

    c = T.CenterCrop((np.array(params['data_aug_shape']) * params['image_scale']).tolist())
    central_crop = lambda x: {'image': c(x)}

    if params['contrastive']:
        data_augment = SimCLRDataTransform(data_augment)
        central_crop = SimCLRDataTransform(central_crop)

    return data_augment, central_crop

def image_transforms(params):
    # Augmentations
    augmentations = []
    if params['rotate']:
        augmentations.append(Rotate(limit=float(params['rotate']), interpolation=3, p=0.9))

    if params['zoom_size'] is not None:
        augmentations.append(RandomScale(scale_limit=0.1, interpolation=3, p=0.9))
        augmentations.append(CenterCrop(*params['zoom_size'], always_apply=True))

    augmentations.append(RandomCrop(*((np.array(params['data_aug_shape']) * params['image_scale']).tolist()), always_apply=True))

    # Pixel based
    augmentations.append(ColorJitter(brightness=params['brightness'], contrast=params['contrast'], saturation=params['saturation'], hue=params['hue'], p=0.9))
    if params['gauss_blur']:
        augmentations.append(Blur(blur_limit=params['gauss_blur'], p=0.5))

    augmentations.append(GaussNoise(var_limit=params['gauss_noise_var_limit'], p=0.9))

    if params['flip']:
        augmentations.append(HorizontalFlip(p=0.5))

    # f = Compose(augmentations)
    f = Compose(augmentations)
    # data_augment = lambda image: torch.Tensor(f(image=image.numpy())['image']).unsqueeze(0)
    data_augment = lambda image: f(image=image.squeeze().numpy())

    # Central crop
    central_crop_f =  CenterCrop(*((np.array(params['data_aug_shape']) * params['image_scale']).tolist()), always_apply=True)
    central_crop_f = ReplayCompose([central_crop_f])
    # central_crop = lambda image: torch.Tensor(central_crop_f(image=image.numpy())['image']).unsqueeze(0)
    central_crop = lambda image: central_crop_f(image=image.squeeze().numpy())

    if params['contrastive']:
        data_augment = SimCLRDataTransform(data_augment)
        central_crop = SimCLRDataTransform(central_crop)

    return data_augment, central_crop

augmentation_suites = {
    'standard': image_transforms,
    'contrastive': contrastive_transforms,
}

if __name__ == "__main__":
    x = torch.randn([1, 256, 256])
    params = {'contrastive': True, 'min_crop_area': 0.25, 'brightness': 0.4, 'contrast': 0.4, 'saturation': 0, 'hue': 0, 'rotate': 10, 'post_rotate_size': [208, 208], 'image_scale': 1, 'data_aug_shape': [192,192]}

    f = torch.nn.Sequential(
            RandomApply(T.ColorJitter(params['brightness'], params['contrast'], params['saturation'], params['hue']), p = 0.8),
            # T.RandomGrayscale(p=0.2),
            T.RandomRotation(params['rotate']),
            T.CenterCrop((np.array(params['post_rotate_size']) * params['image_scale']).tolist()),
            T.RandomHorizontalFlip(),
            # RandomApply(
            #     T.GaussianBlur((3, 3), (0.1, 2.0)),
            #     p = 0.2
            # ),
            T.RandomResizedCrop((np.array(params['data_aug_shape']) * params['image_scale']).tolist(), scale=(params['min_crop_area'], 1.0)),
            # T.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225])),
        )
    y = f(x)
    
    data_augment = lambda x: {'image': f(x)}

    c = T.CenterCrop((np.array(params['data_aug_shape']) * params['image_scale']).tolist())
    central_crop = lambda x: {'image': c(x)}

    if params['contrastive']:
        data_augment = SimCLRDataTransform(data_augment)
        central_crop = SimCLRDataTransform(central_crop)
    y = f(x)

    y = 3