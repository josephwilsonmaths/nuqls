import torch
import pandas as pd
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_dir, transform=None, 
target_transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df['level'][idx]
        img_path = os.path.join(self.img_dir, self.df['image'][idx])
        print(self.df['image'][idx])
        image = Image.open(os.path.join(self.img_dir, img_path + '.jpeg'))
        if self.transform:
            print(image)
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
transform_train = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,4)
        transforms.ToTensor()])

train = CustomImageDataset(r"C:\Users\s4531973\Downloads\trainLabels.csv\trainLabels.csv", r"C:\Users\s4531973\Downloads\train.zip.001\train\train", transform=ToTensor())
f,ax = plt.subplots()

ax.imshow(train[100][0].permute(1,2,0).numpy())

print(torch.mean(train[0][0]))



f.savefig('image1.pdf', format='pdf')