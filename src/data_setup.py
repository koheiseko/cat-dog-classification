import kaggle
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as T
from torchvision import datasets
import numpy as np
import os

def load_dataset(dataset_name, path_dir):
    print("[INFO] Iniciando o download dos dados")
    if not os.path.isdir("data"):
        os.makedirs(name="data", exist_ok=True)
        kaggle.api.dataset_download_files(dataset_name, path=path_dir, unzip=True)
        print("[INFO] Download concluído!")
    else:
        print("[INFO] Os dados já foram baixados")

def get_train_data_transformed(root, image_size, random_flip,random_rotation, mean, std):
    train_transform = T.Compose([
        T.Resize((image_size, image_size)), 
        T.RandomHorizontalFlip(p=random_flip),
        T.RandomRotation(degrees=random_rotation),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    data_transformed = datasets.ImageFolder(root=root, transform=train_transform)

    return data_transformed


def get_test_data_transformed(root, image_size, mean, std):
    test_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    data_transformed = datasets.ImageFolder(root=root, transform=test_transform)    
    
    return data_transformed


def train_test_split(data, test_size:float):
    len_data = len(data)

    qt_test = int(np.floor(len_data * test_size))
    qt_train = len_data - qt_test

    train_data, test_data = random_split(data, [qt_train, qt_test])

    return train_data, test_data


def get_data_loader(train_data=None, valid_data=None, test_data=None, batch_size:int=32, shuffle:bool=True):
    if train_data:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)

    if valid_data:
        valid_loader = DataLoader(dataset=train_data, batch_size=batch_size)

    if test_data:
        test_loader = DataLoader(dataset=train_data, batch_size=batch_size)

    return train_loader, valid_loader, test_loader

