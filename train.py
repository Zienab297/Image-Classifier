import torch
# Imports here
import torchvision
from torchvision import transforms, datasets, models
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default = 'flowers',
                        help='path to image folder')
parser.add_argument('--arch', action='store', type=str, dest='arch', 
                    default = 'vgg16', help='pretrained model')
parser.add_argument('--save_dir', action='store', dest='save',
                     default='checkpoint', help='saving checkpoint')
parser.add_argument('--learning_rate', action='store', dest='learning_rate',
                    default=0.001, help='storing learning rate')
parser.add_argument('--hidden_units', action='store', dest='hidden_units',
                    default=4096, help='number of hidden units')
parser.add_argument('--epochs', action='store', dest='epochs',
                    default=5, help='number of epochs')
parser.add_argument('--gpu', action='store_const', dest='device',
                     const='gpu', default='cpu', help='type of device')

res = parser.parse_args()
data_dir = res.dir
save = res.save
arch = res.arch
learning_rate = res.learning_rate
hidden_units = res.hidden_units
epochs = res.epochs
device = res.device

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {"train": 
                   transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                  "val":
                  transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                  "test":
                  transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

# TODO: Load the datasets with ImageFolder
image_datasets = {"train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
                 "val": datasets.ImageFolder(valid_dir, transform=data_transforms["val"]),
                 "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
              "val": torch.utils.data.DataLoader(image_datasets["val"]),
              "test": torch.utils.data.DataLoader(image_datasets["test"])}


if arch == 'alexnet':
    model = models.alexnet(pretrained = True)
else:
    model = models.vgg16(pretrained = True)

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(25088, hidden_units), 
                           nn.ReLU(),
                           nn.Dropout(0.1),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim=1))



criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def evaluate(model, evalloader, criterion):
    model.to(device)    
    epoch_loss = 0
    epoch_acc = 0
    model.to(device)
    model.eval()

    with torch.no_grad():

        for inputs, labels in tqdm(evalloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            acc = calculate_accuracy(outputs, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(evalloader), epoch_acc / len(evalloader)


def train(model, trainloader, optimizer, criterion):
    model.to(device)
    epoch_loss = 0
    epoch_acc = 0
    model.to(device)
    model.train()
    for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(trainloader), epoch_acc / len(trainloader)


best_valid_loss = float('inf')

for epoch in range(epochs):
    model.to(device)
    train_loss, train_acc = train(model, dataloaders["train"], optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, dataloaders["val"], criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')

    
    print(f'Epoch {epoch+1} / {epochs}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# TODO: Do validation on the test set
model.eval()
true_class = 0
size = 0
with torch.no_grad():
    for inputs, labels in tqdm(dataloaders["test"]):
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        size += labels.size(0)
        true_class += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * true_class / size))

print("Full model: ", model)
print("state dict: ", model.state_dict().keys())
model.class_to_idx = image_datasets['train'].class_to_idx
torch.save({'arch':arch, 'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx}, save+'.pth')
