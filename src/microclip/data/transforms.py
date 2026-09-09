"""Image transforms. Deliberately light: heavy augmentation can break
caption-image correspondence in contrastive training (CLIP uses only
RandomResizedCrop)."""
from torchvision import transforms

# ImageNet stats — fine for from-scratch training too; just be consistent.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
