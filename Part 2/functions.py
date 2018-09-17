import matplotlib.pyplot as plt

import os
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
import json

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

def train(device, model, epochs, criterion, optimizer, print_every, train_loader, test_loader, valid_loader):
    epochs = epochs
    steps = 0

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    test_loss, accuracy = validation(device, model, valid_loader, criterion)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss),
                      "Validation Accuracy: {:.3f}".format(accuracy),
                      "Steps: ", steps)
                running_loss = 0
                                
def validation(device, model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    count = 0
    model.to(device)
    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)

        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

        count += 1

    return test_loss/count, accuracy/count

def save_checkpoint(model, arch, epochs, criterion, optimizer, train_data, save_dir):
    checkpoint = {
    'epochs': epochs,
    'arch': arch,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'class_to_idx': train_data.class_to_idx,
    'fc': model.fc,
    'criterion': criterion
    }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    return None

def load_checkpoint(filepath):
    
    #load checkpoint & rebuild model
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    
    #rebuild model
    loaded_model = models.__getattribute__(arch)(pretrained=True)
    for param in loaded_model.parameters():
        param.requires_grad = False
    loaded_model.fc = checkpoint['fc']
    loaded_model.load_state_dict(checkpoint['state_dict'])
    loaded_model.class_to_idx = checkpoint['class_to_idx']
    
    optimizer = checkpoint['optimizer_dict']
    epochs = checkpoint['epochs']
    criterion = checkpoint['criterion']
    
    return loaded_model, optimizer, criterion, epochs

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return None

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    target_size = 256
    width, height = im.size
    aspect_ratio = width / height
    im.resize((target_size, int(aspect_ratio * target_size)))

    
    # Resize with resize function keeping aspect ratio
    target_size = 256
    width, height = im.size
    ratio = width / height
    if width <= height: 
        im = im.resize((target_size, int(ratio * target_size)))
    else:
        im = im.resize((int(ratio * target_size), target_size))
    
    # Crop Image
    crop_size = 224
    left = (im.size[0] - crop_size)/2
    top = (im.size[1] - crop_size)/2
    right = (im.size[0] + crop_size)/2
    bottom = (im.size[1] + crop_size)/2
    im = im.crop((left, top, right, bottom))
    
    # normalization of colors    
    np_image = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
   
    
    return np_image

def predict(image_path, model, topk, device, json_path):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    with torch.no_grad():
        model.eval()
        
        np_image = process_image(image_path)
        image_tensor = torch.from_numpy(np_image)

        if device == 'cuda':
            model.to('cuda')
            inputs_var = Variable(image_tensor.float().cuda()) 
        else:
            model.to('cpu')
            inputs_var = Variable(image_tensor.float())    
             
        output = model.forward(inputs_var.unsqueeze(0))
        ps = torch.exp(output).topk(topk)
        probs = ps[0].cpu()
        classes = ps[1].cpu()
        
        # Convert indices to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[each] for each in classes.numpy()[0]]
        
        name_list = []
        if isinstance(json_path, type(None)) is False:
            with open(json_path) as file:
                data = json.load(file)
                for flower_id in top_classes:
                     name_list.append(data[str(flower_id)])
                          
    return probs.numpy()[0].tolist(), name_list

def compare_orig_vs_loaded(device, original_model, loaded_model, test_loader, criterion):
    # evaluate trained original model that was used to create checkpoint

    original_model.eval()

    with torch.no_grad():
        test_loss, accuracy = validation(device, original_model, test_loader, criterion)
    print("Test loss original_model= ", test_loss)
    print("Accuracy original_model= ", accuracy)

    #evaluate loaded_model from checkpoint

    loaded_model.eval()

    with torch.no_grad():
        test_loss1, accuracy1 = validation(device, loaded_model, test_loader, criterion)
    print("Test loss loaded_model = ", test_loss1)
    print("Accuracy loaded_model = ", accuracy1)