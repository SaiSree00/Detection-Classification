import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn.functional as F

class MultiClassCelebA(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.image_dir = 'data/'+ image_dir
        self.images = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]  #'%06d'%data_vec[0]+'.jpg'
        path = img_name.replace('.'+img_name.split('.')[-1],'.csv')
        data_vec = pd.read_csv('data/params/'+path)
        data_vec[data_vec>=1] =1
        #.values
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        #label = np.array(data_vec[1:]>-1, dtype=int)
        label = np.array(data_vec.values[0],dtype=int)
        label = label.astype('double')
# =============================================================================
#         if len(image.shape)==2:
#             image = np.expand_dims(image,2)
#             image = np.concatenate((image,image,image),axis=2)
# =============================================================================
        if not image.mode=='RGB':
            image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        
        return sample
        #return image, label

tfms = transforms.Compose([transforms.Resize((256, 256)),
                           transforms.ToTensor()])      
                           

class MultiClassifier(nn.Module):
    def __init__(self):
        super(MultiClassifier, self).__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3), # 3, 256, 256
            nn.MaxPool2d(2), # op: 16, 127, 127
            nn.ReLU(), # op: 64, 127, 127
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3), # 64, 127, 127   
            nn.MaxPool2d(2), #op: 128, 63, 63
            nn.ReLU() # op: 128, 63, 63
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3), # 128, 63, 63
            nn.MaxPool2d(2), #op: 256, 30, 30
            nn.ReLU() #op: 256, 30, 30
        )
        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3), # 256, 30, 30
            nn.MaxPool2d(2), #op: 512, 14, 14
            nn.ReLU(), #op: 512, 14, 14
            nn.Dropout(0.2)
        )
        self.Linear1 = nn.Linear(512 * 14 * 14, 1024)
        self.Linear2 = nn.Linear(1024, 256)
        self.Linear3 = nn.Linear(256, 4)
        
        
    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return F.sigmoid(x)

def check_cuda():
    _cuda = False
    if torch.cuda.is_available():
        _cuda = True
    # else:
    #     _cuda = False
    return _cuda        

is_cuda = check_cuda()

model = MultiClassifier()
#if is_cuda:
   # model.cuda()


model.load_state_dict(torch.load('Model_CLS.pt').state_dict())
model.eval()
img_path = '/home/myid/sm13058/ast/data/images/ave-0000-0002.jpg'

image = Image.open(img_path)

if not image.mode=='RGB':
    image.convert('RGB')
if tfms:
    image = tfms(image)
print(image.shape)
image = torch.unsqueeze(image,0)
out = model(image)
out = out.detach().numpy()
print(out)
