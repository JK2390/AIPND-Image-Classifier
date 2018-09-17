#   Example call:
#    python check.py flowers --arch resnet18 --learning_rate 0.001 --epochs 1 --dropout 0.2 --gpu

import torch
import train
import predict
import functions as fu


original_model, test_loader, criterion = train.main()

#Load pre-trained model from checkpoint
loaded_model, optimizer, criterion, epochs = fu.load_checkpoint('/home/workspace/aipnd-project/checkpoints/checkpoint.pth')

#Set mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('Device is: ', device)

fu.compare_orig_vs_loaded(device, original_model, loaded_model, test_loader, criterion)