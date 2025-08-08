import importlib
# import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_dataset(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    dataset_lib = importlib.import_module(
        '.' + dataset_name, package='dataset')

    dataset_abs = getattr(dataset_lib, dataset_name)
    print(dataset_abs)

    return dataset_abs(**kwargs)


class BaseDataset(Dataset):
    def __init__(self):
        
        self.count = 0
        
    #     basic_transform = [
    #         A.RandomBrightnessContrast(),
    #         A.RandomGamma(),
    #         A.HueSaturationValue()
    #     ]
    #     self.basic_transform = basic_transform    
    #     self.to_tensor = transforms.ToTensor()

    # def augment_training_data(self, image):
    #     H, W, C = image.shape

    #     aug = A.Compose(transforms=self.basic_transform)
    #     augmented = aug(image=image)
    #     image = augmented['image']

    #     return image
