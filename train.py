import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np

import torch.autograd

from skimage import io, color
from skimage.transform import rescale, resize
from torchvision import datasets,transforms
from skimage.color import rgb2lab, rgb2gray, lab2rgb
from skimage import io
import torch.utils.data as data
from skimage.transform import rescale, resize

import sklearn.neighbors as nnb

import random
import argparse

import numpy as np
from PIL import Image
import os

import time

from model import Color_model
from data_loader import LandscapeImageFolder

original_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
])


def main():
    # Create model directory
    model_path = '../model/models'
    batch_size = 2


    if not os.path.exists(model_path):
        os.makedirs(model_path )

    img_dir = './../landscape/'
    train_list = []
    val_list = []
    test_list = []
    with open('./../train.txt','r') as f:
        data = f.readlines()
        for line in data:
            train_list.append(line[:-1])
    with open('./../valid.txt','r') as f:
        data = f.readlines()
        for line in data:
            val_list.append(line[:-1])
    with open('./../test.txt','r') as f:
        data = f.readlines()
        for line in data:
            test_list.append(line[:-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image preprocessing, normalization for the pretrained resnet
    train_set = LandscapeImageFolder(img_dir, train_list, original_transform)
    val_set = LandscapeImageFolder(img_dir, val_list, original_transform)
    test_set = LandscapeImageFolder(img_dir, test_list, original_transform)    

    # Build data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 1)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 1)


    # Build the models
    model = (Color_model()).to(device, torch.float)
    #model.load_state_dict(torch.load('../model/models/model-171-216.ckpt'))
#    encode_layer=NNEncLayer()
#    boost_layer=PriorBoostLayer()
#    nongray_mask=NonGrayMaskLayer()
    weights = np.load("./../weights.npy")
    weights = torch.from_numpy(weights)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduce=True, weight=weights).to(device, torch.float)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = 1e-3)
    
    num_epochs = 10
    log_step = 1
    save_step = 80
    # Train the models

    since = time.time()

    model.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        losses = []
        try:
            for i, (images, img_ab) in enumerate(train_loader):
                try:
                    tmp = time.time()   
                    # Set mini-batch dataset
                    images = images.unsqueeze(1).to(device, torch.float)
                    targets=img_ab.to(device,torch.long)
#                    img_ab = img_ab#.float()
#                    encode,max_encode=encode_layer.forward(img_ab)

#                    targets=torch.Tensor(img_ab).float().to(device, torch.float)

#                    boost=torch.Tensor(boost_layer.forward(encode)).to(device, torch.float)
#                    mask=torch.Tensor(nongray_mask.forward(img_ab)).to(device, torch.float)

#                    boost_nongray=boost*mask
                    outputs = model(images)#.log()
#                    output=outputs[0].cpu().data.numpy() 

#                    out_max=np.argmax(outputs ,axis=0)
                    loss = criterion(outputs,targets)
#                    loss = (criterion(outputs,targets)*(boost_nongray.squeeze(1))).mean()

                    model.zero_grad()
                
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    duration = time.time()-tmp

                    # Print log info
                    if i % log_step == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, {:.4f}s'.format(epoch, num_epochs, i, total_step, losses[-1], duration))

                    # Save the model checkpoints
                    if (i + 1) % save_step == 0:
                        torch.save(model.state_dict(), os.path.join('../model/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                except:
                    pass
            print('Epoch [{}/{}], Avg Loss: {:.4f}'.format(epoch, num_epochs, np.mean(losses)))
        except:
            pass

if __name__ == '__main__':
    main()
