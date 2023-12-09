import torch.nn as nn
import toch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class DataAugmentation:
    """
    We follow the data augmentations of BYOL:
    - Horizontal flip
    - Color jittering
    - Gaussian blur
    - Solarization
    - Multi-crop
    
    dataset_mean/dataset_std: Mean and Std values of the dataset (default values are for ImageNet) 
    """
    
    def __init__(
        self, 
        img_size: int,
        global_crop_min_size: int, 
        global_crop_max_size: int, 
        local_crop_min_size: int, 
        local_crop_max_size: int, 
        nb_local_crops: int, 
        dataset_mean: list = [0.485, 0.456, 0.406], 
        dataset_std: list = [0.229,0.224,0.225]):

        self.nb_local_crops = nb_local_crops
        
        flip = transforms.RandomHorizontalFlip(p=0.5)
        color_jitter = transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8)
        
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=dataset_mean, std=dataset_std)])
        
        # Global crops
        self.apply_global_crop_1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(global_crop_min_size, global_crop_max_size), interpolation=Image.BICUBIC),
            flip,
            color_jitter,
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            normalize
        ])
        
        self.apply_global_crop_2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(global_crop_min_size, global_crop_max_size), interpolation=Image.BICUBIC),
            flip,
            color_jitter,
            transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]), p=0.1),
            transforms.RandomApply(nn.ModuleList([transforms.RandomSolarize(threshold=170)]), p=0.2),
            normalize
        ])
        
        # Local crops
        self.apply_local_crop = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(local_crop_min_size, local_crop_max_size), interpolation=Image.BICUBIC), # keep the same size as global crops (upscaling) so no tensor shape problems
            flip,
            color_jitter,
            transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]), p=0.5),
            normalize
        ])
        
        def __call__(self, image):
            global_crop_1 = self.apply_global_crop_1(image)
            global_crop_2 = self.apply_global_crop_2(image)
            local_crops = [self.apply_local_crop(image) for _ in range(self.nb_local_crops)]
            
            return [global_crop_1, global_crop_2] + local_crops
            
        