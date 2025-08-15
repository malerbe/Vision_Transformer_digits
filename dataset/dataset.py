import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import random

class ImageCaptchaDataset(Dataset):
    def __init__(self, root, img_h=32, img_w=128, vocab='0123456789', transform=None, train=True):
        """
        Args:
            folder (str): chemin du dossier contenant les images
            img_h, img_w : dimensions de resize spatiale
            vocab (str): caractères possibles (doivent couvrir tous les labels)
            transform: transform PIL -> tensor
        """
        self.train = train
        self.folder = root
        self.img_h = img_h
        self.img_w = img_w
        self.vocab = vocab
        self.vocab2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2vocab = {i: c for c, i in self.vocab2idx.items()}
        self.files = sorted([f for f in os.listdir(self.folder) if f.lower().endswith(('.png'))])
        random.shuffle(self.files)
        if train:
            self.files = self.files[:(len(self.files)//100)*80]
        else:
            self.files = self.files[(len(self.files)//100)*80:]
            

        self.classes = vocab
        
        # By default: center crop and normalize images
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        print("TRANSFORM: ", self.transform)

    def __getitem__(self, idx):
        file = self.files[idx]
        img_path = os.path.join(self.folder, file)
        # Ouvre et traite l'image
        img = Image.open(img_path)
        # img = img.point(lambda x: 255 if x > 0 else 0)
        img = img.convert('RGB')
        
       
        img = self.transform(img)
        
        # Prend le nom avant l'extension comme label ex: '12345.png' -> '12345'
        label_str = os.path.splitext(file)[0].split("_")[0]
        label = [self.vocab2idx[c] for c in label_str]
        label = torch.tensor(label, dtype=torch.long)
        label = label[0]
        return img, label

    def __len__(self):
        return len(self.files)
