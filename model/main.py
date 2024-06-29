import train
import dataloader
import numpy as np
import torch
from torchvision.models import mobilenet_v2
from tqdm import tqdm

# Initializes and trains a mobilenet_v2 model on a set of images
def main():
    print("Making dataset...")
    # Initialize training and validation sets from image folder
    data = dataloader.MushroomDataset('images', batch_size= 8)
    print("Training...")
    # Intialize trainer
    trainer = train.Trainer()
    epochs = 100
    lr = 1e-3
    # Set device to GPU if available, otherwise use cpu
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize model and optimizer
    nn_model = mobilenet_v2(num_classes=1382).to(device)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    # Train for n epochs
    for _ in tqdm(range(epochs)):
        trainer.train_epoch(data, nn_model, optimizer)
        trainer.val_epoch(data, nn_model)
        # Save model
        torch.save(nn_model.state_dict(), 'mushroom_id.pt')

if __name__ == "__main__":
    main()