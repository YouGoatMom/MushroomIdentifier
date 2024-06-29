from tqdm import tqdm
import torch
import torchvision

class Trainer():

    def __init__(self):
        self.lr = 1e-3
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)
        # loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.transform = torchvision.transforms.Compose([torchvision.transforms.RandomRotation((-15, 15)),
                                                    torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.RandomHorizontalFlip(), 
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def train_epoch(self, data_loader, model, optimizer):
        model.train()
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        for i, data in tqdm(enumerate(data_loader.train_dataloader), total=len(data_loader.train_dataloader)):
            counter += 1
            image, labels = data
            image = image.to(self.device)
            labels = labels.to(self.device)
            image = self.transform(image)
            optimizer.zero_grad()
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = self.loss_fn(outputs, labels)
            train_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            # print("Example output: ")
            # print(data_loader.dataset.classes[labels[0].item()])
            # print(data_loader.dataset.classes[preds[0]])
            train_running_correct += (preds == labels).sum().item()
            # backpropagation
            loss.backward()
            # update the optimizer parameters
            optimizer.step()
        
        # loss and accuracy for the complete epoch
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(data_loader.train_dataloader.dataset))
        print("Epoch Loss: ", epoch_loss, "Epoch Accuracy", epoch_acc)
        return epoch_loss, epoch_acc
    
    def val_epoch(self, data_loader, model):
        running_vloss = 0.0
        running_vcorrect = 0
        counter = 0
        model.eval()
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(data_loader.test_dataloader), total=len(data_loader.test_dataloader)):
                counter += 1
                vinputs, vlabels = vdata
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)
                vinputs = self.transform(vinputs)
                voutputs = model(vinputs)
                _, preds = torch.max(voutputs.data, 1)
                running_vcorrect += (preds == vlabels).sum().item()
                vloss = self.loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
        avg_vloss = running_vloss / counter
        avg_acc = 100 * running_vcorrect / len(data_loader.test_dataloader.dataset)
        print("Val Epoch Loss: ", avg_vloss, "Val Epoch Accuracy", avg_acc)