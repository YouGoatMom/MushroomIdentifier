import train
import dataloader
import model

def main():
    print("Making dataset...")
    data = dataloader.MushroomDataset('data')
    print("Creating model...")
    mush_model = model.Model()
    print("Training...")
    trainer = train.Trainer()
    trainer.train(data, mush_model, 10)

if __name__ == "__main__":
    main()