import torch
import torchvision
import torchvision.transforms as transforms
import math

from DataAugmentation import DataAugmentation

class Config:
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


def get_train_test_dataloaders(config: dict):
    data_aug = DataAugmentation(
        config.img_size,
        config.global_crop_ratio,
        config.local_crop_ratio,
        config.nb_local_crops,
        config.dataset_means,
        config.dataset_stds
    )
    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    dataset_train_plain = torchvision.datasets.CIFAR10(
        root=config.train_dataset_plain_path,
        train=True,
        transform=transform_plain,
        download=True
    )
    train_dataloader_plain = torch.utils.data.DataLoader(
        dataset_train_plain,
        batch_size=config.batch_size_eval,
        num_workers=4,
        drop_last=False
    )

    dataset_train_aug = torchvision.datasets.CIFAR10(
        root=config.train_dataset_aug_path,
        train=True,
        transform=data_aug,
        download=True
    )
    train_dataloader_aug = torch.utils.data.DataLoader(
        dataset_train_aug,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    dataset_test = torchvision.datasets.CIFAR10(
        root=config.test_dataset_path,
        train=False,
        transform=transform_plain,
        download=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        num_workers=4,
        drop_last=False
    )
    
    return train_dataloader_plain, train_dataloader_aug, test_dataloader



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def clip_gradients(model, clip_value=2.0):
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip_value / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)