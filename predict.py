import torch

# Imports here
import torchvision
from torchvision import transforms, datasets, models
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import json
import argparse
'''parser is used to optimize the command files by adding arguments 
* --dir argument is used to identify the image folder path
* --save_dir for saving the checkpoint file
* --top_k is for storing the classes that have the highest probabilities
* --class_names is for storing the acual names of the flowers using the file cat_to_name.json'''
parsers = argparse.ArgumentParser()
parsers.add_argument('--dir', type=str, default = 'flowers', dest='data_dir',
                        help='path to image folder')
parsers.add_argument('--save_dir', action='store', dest='save',
                     default='checkpoint')
parsers.add_argument('--top_k', dest="top_k", default=5)
parsers.add_argument('--class_names', action="store", 
                     dest="class_names", default='cat_to_name.json')
parsers.add_argument('--hidden_units', action='store', dest='hidden_units',
                    default=500, help='number of hidden units')

var = parsers.parse_args()
data_dir = var.data_dir
save = var.save
top_k = var.top_k
class_names = var.class_names
hidden_units = var.hidden_units

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

checkpoint = torch.load(save+'.pth') #loading checkpoint
checkpoint.keys()

model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(nn.Linear(25088, hidden_units), 
                           nn.ReLU(),
                           nn.Dropout(0.1),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim=1))

model.class_to_idx = checkpoint['class_to_idx']
model.load_state_dict(checkpoint['state_dict'])


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    size = 224, 224
    img = Image.open(image)
    img.thumbnail(size)

    np_image = np.array(img)
    np_image = np_image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

x= process_image('flowers/train/1/image_06735.jpg')

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1, 2, 0))    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

imshow(x)
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.unsqueeze_(0)
    result = model.forward(image)
    exp = torch.exp(result)
    probs, labels = exp.topk(5)
    return probs, labels

predict('flowers/train/1/image_06735.jpg', model, top_k)

image = 'flowers/train/1/image_06735.jpg'
img = process_image(image)
imshow(img)

probs, labels = predict(image, model)
probs = probs.tolist()[0]
labels = labels.tolist()[0]
flowers = [i for i in labels]
flower_cat = []
for i in flowers:
    for key, value in checkpoint['class_to_idx'].items():
        if value == i:
            flower_cat.append(key)
flower_name = [cat_to_name[i] for i in flower_cat]
ax = plt.subplot(2,1,2)
ax.barh(flower_name, probs)
plt.show()