import torch
import torchvision
from torch.utils.data import DataLoader
import json

class MushroomDataset():
    def __init__(self, dir, batch_size):
        transform = torchvision.transforms.Compose([torchvision.transforms.RandomRotation((-15, 15)),
                                                    torchvision.transforms.RandomResizedCrop((224, 224)),
                                                    torchvision.transforms.RandomHorizontalFlip(), 
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.dataset = torchvision.datasets.ImageFolder(dir, transform=transform)
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        # Save index - class dictionary as JSON for API
        with open("classes_dict.json", "w") as outfile: 
            json.dump(self.dataset.classes, outfile)