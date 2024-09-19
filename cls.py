import torchvision
from glob import glob
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.utils import make_grid
import time
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn.functional as F
from pprint import pprint

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
                           
train_dl = MultiClassCelebA('', 'train', transform = tfms)
valid_dl = MultiClassCelebA('', 'val', transform = tfms)

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
if is_cuda:
    model.cuda()

train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = 32, num_workers = 1)
valid_dataloader = torch.utils.data.DataLoader(valid_dl, shuffle = True, batch_size = 32, num_workers = 1)  

def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy()/len(original)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

def fit_model(epochs, model, dataloader, phase = 'training', volatile = False):
    
    pprint("Epoch: {}".format(epochs))

    if phase == 'training':
        model.train()
        
    if phase == 'validataion':
        model.eval()
        volatile = True
        
    running_loss = []
    running_acc = []
    b = 0
    for i, data in enumerate(dataloader):
        

        inputs, target = data['image'].cuda(), data['label'].float().cuda()
        
        inputs, target = Variable(inputs), Variable(target)
        
        if phase == 'training':
            optimizer.zero_grad()
            
        ops = model(inputs)
        
        acc_ = []
        for i, d in enumerate(ops, 0):
           
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d))
            acc_.append(acc)

        loss = criterion(ops, target)
                
        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        b += 1
       
        if phase == 'training':
            
            loss.backward()
        
            optimizer.step()
            
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    

    pprint("{} loss is {} ".format(phase,total_batch_loss))
    pprint("{} accuracy is {} ".format(phase, total_batch_acc))
    
    return total_batch_loss, total_batch_acc

from  tqdm import tqdm 

trn_losses = []; trn_acc = []
val_losses = []; val_acc = []
for i in tqdm(range(1, 50)):
    trn_l, trn_a = fit_model(i, model, train_dataloader)
    val_l, val_a = fit_model(i, model, valid_dataloader, phase = 'validation')
    trn_losses.append(trn_l); trn_acc.append(trn_a)
    val_losses.append(val_l); val_acc.append(val_a)

torch.save(model, "Model_CLS.pt")       
