import train
import dataloader
import model
import torch
from torchvision.models import mobilenet_v2
from tqdm import tqdm

def main():
    print("Making dataset...")
    # Initialize training and validation sets from data folder
    data = dataloader.MushroomDataset('data', batch_size= 8)
    print("Training...")
    # Intialize trainer
    trainer = train.Trainer()
    epochs = 500
    lr = 1e-3
    # Set device to GPU if available, otherwise use cpu
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize model and optimizer
    nn_model = mobilenet_v2(num_classes=215).to(device)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    # Train for n epochs
    for _ in tqdm(range(epochs)):
        trainer.train_epoch(data, nn_model, optimizer)
        trainer.val_epoch(data, nn_model)
    # Save model
    torch.save(nn_model.state_dict(), 'mushroom_identifier.pt')

if __name__ == "__main__":
    main()