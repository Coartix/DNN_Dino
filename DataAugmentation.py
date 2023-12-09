import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class DataAugmentation:
    """
    We follow the data augmentations of BYOL:
    - Horizontal flip
    - Color jittering*
    - Gaussian blur
    - Solarization
    - Multi-crop
    
    dataset_mean/dataset_std: Mean and Std values of the dataset
    """
    
    def __init__(
        self, 
        img_size: int,
        global_crop_ratio: tuple,
        local_crop_ratio: tuple, 
        nb_local_crops: int,
        dataset_means: list, 
        dataset_stds: list):

        self.nb_local_crops = nb_local_crops
        
        flip = transforms.RandomHorizontalFlip(p=0.5)
        color_jitter = transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8)
        
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=dataset_means, std=dataset_stds)])
        
        blur_kernel_size = int(img_size * 0.1)
        
        # Global crops
        self.apply_global_crop_1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crop_ratio, interpolation=Image.BICUBIC),
            flip,
            color_jitter,
            transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.1, 2.0)),
            normalize
        ])
        
        self.apply_global_crop_2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crop_ratio, interpolation=Image.BICUBIC),
            flip,
            color_jitter,
            transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.1, 2.0))]), p=0.1),
            transforms.RandomApply(nn.ModuleList([transforms.RandomSolarize(threshold=170)]), p=0.2),
            normalize
        ])
        
        # Local crops
        self.apply_local_crop = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=local_crop_ratio, interpolation=Image.BICUBIC), # keep the same size as global crops (upscaling) so no tensor shape problems
            flip,
            color_jitter,
            transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.1, 2.0))]), p=0.5),
            normalize
        ])
        
    def __call__(self, image):
        global_crop_1 = self.apply_global_crop_1(image)
        global_crop_2 = self.apply_global_crop_2(image)
        local_crops = [self.apply_local_crop(image) for _ in range(self.nb_local_crops)]
        
        return [global_crop_1, global_crop_2] + local_crops
            
        