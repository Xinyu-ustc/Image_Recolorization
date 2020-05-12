import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from data_loader import LandscapeImageFolder, TestImageFolder
from model import Color_model
import time
from PIL import Image
from torchsummary import summary
import pickle
from skimage.color import rgb2lab, rgb2gray,lab2rgb
import matplotlib.pyplot as plt

original_transform = transforms.Compose([
    transforms.Resize((224,224)),
#    transforms.RandomCrop(224),
#    transforms.RandomHorizontalFlip(),
])


def main():
    # Create model directory
    model_path = '../model/models'
    batch_size = 2


    if not os.path.exists(model_path):
        os.makedirs(model_path )

    img_dir = './../landscape_imgs'
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
    with open('./../landscape_imgs/test.txt','r') as f:
        data = f.readlines()
        for line in data:
            test_list.append(line[:-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image preprocessing, normalization for the pretrained resnet
#    train_set = LandscapeImageFolder(img_dir, train_list, original_transform)
#    val_set = LandscapeImageFolder(img_dir, val_list, original_transform)
    test_set = TestImageFolder(img_dir, test_list, original_transform)    

    # Build data loader
#    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 1)
#    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 1)

    cclass= list(np.load("./../classes.npy"))
    with open("./../kmeans.pkl",'rb') as f:
        data = pickle.load(f)

    keylist = np.zeros((len(data.keys()),3))
    cnt = 0 
    for key in data.keys():
        tmp = key.split(',')
        keylist[cnt, 0] = int(tmp[0])
        keylist[cnt, 1] = int(tmp[1])        
        keylist[cnt, 2] = int(tmp[2])
        cnt = cnt+1
    
    # Build the models
    model = (Color_model()).to(device, torch.float)
    
    model.load_state_dict(torch.load('./anneal/model1-30.ckpt',map_location=torch.device('cpu')))


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduce=True).to(device, torch.float)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = 1e-3)
    
    num_epochs = 10
    log_step = 10
    save_step = 80
    # Train the models

    since = time.time()

    model.eval()
#    total_step = len(train_loader)

    for i, (images, img_ab, fname) in enumerate(test_loader):
                    # Set mini-batch dataset
        images = images.unsqueeze(1).to(device, torch.float)

        targets=img_ab.to(device,torch.long)

        outputs = model(images)#.log()

        output_ab = np.argmax(outputs.detach().numpy(), axis = 1)

        output_ab = output_ab.repeat(4, axis=1).repeat(4, axis=2)

        out_img = np.zeros((224,224,3))

        qq = np.zeros(3)

        tmp_img = np.zeros((224,224,3))
        for i in range(224):
            for j in range(224):
#                qq[0] = round((images[0, 0, i, j].numpy()).item())+50
#                tmp = cclass[output_ab[0,i,j]].split(',')
#                qq[1] = int(tmp[0])
#                qq[2] = int(tmp[1])
                tmp_img[i,j,0] = round((images[0, 0, i, j].numpy()).item())+50
                tmp = cclass[output_ab[0,i,j]].split(',')
                tmp_img[i,j,1] = int(tmp[0])
                tmp_img[i,j,2] = int(tmp[1])
#                index =np.argmin(np.sum((keylist-qq)**2,axis=1))
#                value = data[str(int(keylist[index][0]))+','+str(int(keylist[index][1]))+','+str(int(keylist[index][2]))]   
#                out_img[i,j,:] = value/255.0                
##TODO:
##map each pixel in tmp_img to nearest key in data.keys() and use that key's value as the rgb value 
#in stead of directly using lab2rgb as it may cause some invalid rgb value.
##Change resolution from 56*56 to 224*224/256*256

#                while(str(int(qq[0]))+','+str(int(qq[1]))+','+str(int(qq[2])) not in data.keys()):
#                    qq[0] = qq[0]-1
#                value = data[str(int(qq[0]))+','+str(int(qq[1]))+','+str(int(qq[2]))]   
#                out_img[i,j,:] = value/255.0
        out_img = lab2rgb(tmp_img)
        im = Image.fromarray((out_img*255).astype('uint8')).convert('RGB')
        im_save_path = './results/1_30'#'./test1'#'./../landscape_imgs/recolor'        
        im.save(os.path.join(im_save_path,fname[0]))
#        im.save(os.path.join('./../landscape_imgs/recolor',fname[0]))
#    print('Epoch [{}/{}], Avg Loss: {:.4f}'.format(epoch, num_epochs, np.mean(losses)))


if __name__ == '__main__':
    main()
