import numpy as np
import os
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
from skimage import io

class Apple2OrangeDataset(Dataset):
    """Apple2Orange dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attr = pd.read_csv(csv_file)
        #self.attr.index = self.attr.index.map(lambda x : str(x).split(",")[0])
        self.attr = self.attr.set_index("image_id")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = os.listdir(self.root_dir)
        img_name = os.path.join(self.root_dir, imgs[idx])

        image = io.imread(img_name)
        key = imgs[idx][:-4]
        attrs = self.attr["domain"].loc[[key]]
        try:
            attrs = np.array([attrs.iloc[0]])
        except:
            print(img_name)

        if self.transform:
            image =  self.transform(image)

        attrs = list(map(lambda x : 1 if x == "A (Summer)" else 0, attrs))
        attrs = torch.tensor(attrs)

        return image, attrs

class CelebADataset(Dataset):
    """CelebA dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. 40 attributes
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attr = pd.read_csv(csv_file, sep='\s+')
        #self.attr = self.att
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = os.listdir(self.root_dir)
        img_name = os.path.join(self.root_dir, imgs[idx])

        image = io.imread(img_name)
        attrs = self.attr.loc[[imgs[idx]]]
        try:
            attrs = np.array([attrs.iloc[0]])
        except:
            print(img_name)

        if self.transform:
            image =  self.transform(image)

        return image, attrs

class SkyFinderDataset(Dataset):
    """SkyFinder dataset."""

    def __init__(self, csv_file, root_dir, attrs, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            attrs (list): attributes to select
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attr = pd.read_csv(csv_file)
        #self.attr = self.att
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = os.listdir(self.root_dir)
        img_name = os.path.join(self.root_dir, imgs[idx])

        image = io.imread(img_name)
        attrs = self.attr.loc[self.attr["Filename"] == imgs[idx]]
        try:
            attrs = np.array([attrs["night"].iloc[0]])
        except:
            print(img_name)

        if self.transform:
            image =  self.transform(image)

        return image, attrs

def population_mean_norm(path): #Utility function to normalize input data based on mean and standard deviation of the entire dataset
    train_dataset1 = torchvision.datasets.ImageFolder(
            root=path,
            transform=transforms.Compose([
                    transforms.ToTensor()
                    ])
        )

    dataloader = torch.utils.data.DataLoader(train_dataset1, batch_size=4096, shuffle=False, num_workers=4)

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for data,label in dataloader:
        numpy_image = data.numpy()
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    return pop_mean, pop_std0


def show(img, title, epoch, orig): #Utility function to show figures and plots
    npimg = img.numpy()
    plt.figure()
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig('results/results_'+orig+"_"+str(epoch)+'.png')

