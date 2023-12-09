import torch
import torchvision

from DataAugmentation import DataAugmentation

class Config:
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


def get_train_test_dataloaders(config: dict):
    data_aug = DataAugmentation(
        config.img_size,
        config.global_crop_min_size,
        config.global_crop_max_size,
        config.local_crop_min_size,
        config.local_crop_max_size,
        config.nb_local_crops,
        config.dataset_means,
        config.dataset_stds
    )

    dataset_train = torchvision.datasets.CIFAR10(
        root=config.train_dataset_path,
        train=True,
        transform=data_aug,
        download=True
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root=config.test_dataset_path,
        train=False,
        transform=data_aug,
        download=True
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return train_dataloader, test_dataloader