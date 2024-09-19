import os 
import glob
import pandas as pd
import shutil
import random

images = os.listdir('data/images')

labels = os.listdir('data/params')


for i in images:
    lb_path = 'data/params/'+i.replace('.'+i.split('.')[-1],'.csv')
    if not os.path.exists(lb_path):
        try:
            os.remove('data/images/'+i)
        except Exception as e:
            print(e)
            pass
        print(lb_path)

for i in ['train','val','labels']:
    try:
        os.system('mkdir -p data/'+i)
    except Exception as e:
        print(e)
        pass
    

random.shuffle(images)

n = len(images)

for i in range(0,int(n*0.8)):
    try:
        shutil.copy('data/images/'+images[i],'data/train/')
    except Exception as e:
        print(e)
        continue


for i in range(0,int(n*0.8)):
    try:
        shutil.copy('data/images/'+images[i],'data/val/')
    except Exception as e:
        print(e)
        continue


for i in os.listdir('data/params'):
    try:
        ls = pd.read_csv('data/params/'+i)
        ls[ls==0] = -1
        ls.to_csv('data/labels/'+i,index=False)
    except Exception as e:
        print(e)
        pass
