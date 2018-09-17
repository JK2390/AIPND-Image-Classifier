#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/aipnd-project/train.py
#                                                                             
# PROGRAMMER: Julian Kleinz
# DATE CREATED: 09/11/2018                                  
# REVISED DATE: 09/17/2018 - 2nd submission

# PURPOSE: Predict flower name and show probability of most likely flower names.
#          Load saved model checkpoint and mapping file containing real categories.
# 
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py 
#             image_path <path of image to be analyzed>
#             checkpoint_dir <path of checkpoint with pre-trained model to be used for prediction>
#             --top_k <number of top K most likely classes> 
#             --category_names <mapping of categories to real names from .json file>
#             --gpu <enable cuda for inference>
#
#   Example call:
#    python predict.py /home/workspace/aipnd-project/flowers/test/18/image_04292.jpg checkpoints --gpu --top_k 3 --category_names cat_to_name.json
##

import torch
import argparse
import functions as fu

def main():
    #Set up argument parser for console input
    parser = argparse.ArgumentParser(description='Predict category of flower')
    parser.add_argument('image_path', help='path of image to be analyzed')
    parser.add_argument('checkpoint_dir', help='directory containing /checkpoint.pth with pre-trained model to be used for prediction')
    parser.add_argument('--top_k', help='number of top K most likely classes', default=1, type=int)
    parser.add_argument('--category_names', help='Select JSON file')
    parser.add_argument('--gpu', help='Enable GPU', action='store_true')

    args = parser.parse_args()

    #Load pre-trained model from checkpoint
    loaded_model, optimizer, criterion, epochs = fu.load_checkpoint(args.checkpoint_dir + '/checkpoint.pth')

    #Set mode
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==True else 'cpu') 
    print('Device is: ', device)

    #Inference calculation
    probs, classes = fu.predict(args.image_path, loaded_model, args.top_k, device, args.category_names)
    print(probs)
    print(classes)

if __name__ == "__main__":
    main()