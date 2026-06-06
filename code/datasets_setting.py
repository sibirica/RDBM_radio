import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils 

from PIL import Image
from torch import nn
import imageio.v3 as imageio
import cv2
import numpy as np
import random
import json

import torch.nn.functional as F

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def default(val, d):
    if exists(val) and (val is not None):
        return val
    return d() if callable(d) else d

def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

degradtion_cache = ['rain','haze','light','shadow','blur']

class RSDataset(Dataset):
    def __init__(self, root_dir, task_folder, sub_folder = None, mode = 'train', crop_size = 128):
        super().__init__() 
        assert mode in ['train','test']
        assert task_folder in degradtion_cache
        self.root_path = root_dir
        self.task_path = os.path.join(root_dir,task_folder)
        if sub_folder is not None:
            self.sub_datasets = [sub_folder]
        else:
            self.sub_datasets = os.listdir(self.task_path) 
        self.mode = mode
        print('Datasets',self.sub_datasets)
        self.crop_size = crop_size  
        
        self.max_choices = 4

        if mode == 'train':
            self.skip_datasets = [] 

        if mode == 'test':
            self.skip_datasets = [] 
            
        self.input_path, self.target_path = [], []     

        for sub_dataset in self.sub_datasets:   
            # print(sub_dataset,os.listdir(self.task_path))
            assert sub_dataset in os.listdir(self.task_path)
            if sub_dataset in self.skip_datasets:
                continue 
            else:
                json_dir = os.path.join(self.task_path,sub_dataset,'meta')
                if not os.path.exists(json_dir):
                    print(json_dir)
                    exit('No such json directory')
                json_path = os.path.join(json_dir,'{}.json'.format(mode))
                json_file = self.load_json(json_path) 
                input_list, target_list = [], []
                for item in json_file:
                    input_ , target_= item['input'] , item['target'] 
                    input_ = os.path.join(self.task_path,input_)
                    target_ = os.path.join(self.task_path,target_) 
                    input_list.append(input_)
                    target_list.append(target_)
            
            self.input_path = self.input_path + input_list
            self.target_path = self.target_path + target_list      
        
        assert len(self.input_path) == len(self.target_path) 
        
        self.transform = T.ToTensor()
 
    def __len__(self):
        assert len(self.target_path) > 0, f"meta_list is empty, check file: {self.target_path}"
        return len(self.target_path)
    
    def __getitem__(self, index):   
        input_file_path = self.input_path[index]   
        target_file_path = self.target_path[index] 
            
        im_degrade = self.load_image(input_file_path)
        im_clean = self.load_image(target_file_path)   
        
        if not self.check_size(im_degrade,im_clean):
            print(target_file_path, im_clean.shape,'--->',input_file_path, im_degrade.shape) 
            exit('Unmatched image sizes')
            
        im_path = target_file_path # os.path.basename(target_file_path)
        
        if self.mode == 'train':
            im_degrade,im_clean = self.random_crop_size(im_degrade,im_clean,self.crop_size)  
            
        if self.transform is not None:  
            im_degrade = self.transform(im_degrade) 
            im_clean = self.transform(im_clean)
        
        if self.mode == 'test':
            im_degrade = self._padding(im_degrade)
        
        return im_degrade,im_clean,im_path
    
    def check_size(self,input,target):
        flag = 1
        if isinstance(input, list):
            for item in input:
                if item.shape != target.shape:
                    flag = 0
                    return flag
        else:
            if input.shape != target.shape:
                flag = 0
                return flag
        return flag
        
    def load_image(self,path,data_range=255.0):
        if isinstance(path, list):
            sample = [np.clip(imageio.imread(x)[:,:,[0,1,2]],0,data_range).astype('float32') / data_range for x in path] 
        else:
            sample = np.clip(imageio.imread(path)[:,:,[0,1,2]],0,data_range).astype('float32') / data_range
        return sample
    
    def load_json(self,json_path):
        data = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        data.append(obj)
                except json.JSONDecodeError:
                    bad_lines += 1
                    print(f"{line_num} line in JSON")
        return data

    def resize_shape(self, image, short_side_length = 256): 
        if isinstance(image, list): 
            imageset = []
            for item in image:           
                oldh, oldw, _ = item.shape[0],item.shape[1],item.shape[2]
                if min(oldh, oldw) < short_side_length:
                    scale = short_side_length * 1.0 / min(oldh, oldw)
                    newh, neww = oldh * scale, oldw * scale
                    imageset.append(cv2.resize(item,(int(neww + 0.5),int(newh + 0.5))))
                else:
                    imageset.append(item)
            return imageset
        else:
            oldh, oldw, _ = image.shape[0],image.shape[1],image.shape[2]
            if min(oldh, oldw) < short_side_length:
                scale = short_side_length * 1.0 / min(oldh, oldw)
                newh, neww = oldh * scale, oldw * scale
                image = cv2.resize(image,(int(neww + 0.5),int(newh + 0.5))) 
            return image
    
    def random_crop_size(self,imageA,imageB,crop_size):
        imageA = self.resize_shape(imageA,crop_size)
        imageB = self.resize_shape(imageB,crop_size)
         
        if not self.check_size(imageA,imageB): 
            exit('Unmatched image sizes')
            
        h,w,_ = imageB.shape
        h_start,w_start = np.random.randint(0,h-crop_size+1),np.random.randint(0,w-crop_size+1)
        if isinstance(imageA, list):
            imageA_crop = [x[h_start:h_start+crop_size,w_start:w_start+crop_size,:] for x in imageA]
        else:
            imageA_crop = imageA[h_start:h_start+crop_size,w_start:w_start+crop_size,:]
            
        if isinstance(imageB, list):  
            imageB_crop = [x[h_start:h_start+crop_size,w_start:w_start+crop_size,:] for x in imageB]
        else:  
            imageB_crop = imageB[h_start:h_start+crop_size,w_start:w_start+crop_size,:]
        
        return imageA_crop,imageB_crop
    
    def _padding(self, data, pading_shape = 32): 
        h, w = data.shape[-2],data.shape[-1]
        
        if h % pading_shape == 0:
            pad_h = 0
        else:
            pad_h = pading_shape - h % pading_shape
        if w % pading_shape == 0:
            pad_w = 0
        else:
            pad_w = pading_shape - w % pading_shape

        pad_top = 0
        pad_bottom = pad_h
        pad_left = 0
        pad_right = pad_w 
        data = F.pad(data, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)  
        return data 
    
if __name__ == '__main__':
    print('Hello World')
    set_seed(11)
    N = 100
    root_dir = ""
    for detyep in ['rain','haze','light','shadow','blur']:  
        trainset = RSDataset(root_dir = root_dir,task_folder = detyep, sub_folder = None, mode = 'train',)
        print(len(trainset)) 
        dataloader = DataLoader(trainset,  batch_size=1, pin_memory=True, shuffle = True) 
        for i, (lq, hq, paths) in enumerate(dataloader):
            if i < N:  
                print(f"{detyep}   {paths[0]}: hq shape={hq.shape}, lq shape={lq.shape}")
            else:
                break
        
        testset = RSDataset(root_dir = root_dir,task_folder = detyep, sub_folder = None, mode = 'test',)
        print(len(testset)) 
        dataloader = DataLoader(testset,  batch_size=1,  pin_memory=True, shuffle = True) 
        for i, (lq, hq, paths) in enumerate(dataloader):
            if i < N:  
                print(f"{detyep}   {paths[0]}: hq shape={hq.shape}, lq shape={lq.shape}")
            else:
                break
