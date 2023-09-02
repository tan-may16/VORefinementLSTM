
from collections import Counter
import os
# import re
from PIL import Image
import torch
from torch.utils.data import Dataset
# import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np


class KITTIDataset(Dataset):
    def __init__(self, image_dir, image_filename_pattern, odom_file, seq_length = 50, gt_file = None, length= 224, width = 224):
        
        self._image_dir = image_dir
        self.image_paths = [os.path.join(self._image_dir, f) for f in os.listdir(self._image_dir)]
        self._image_filename_pattern = image_filename_pattern
        self.length = length ## Resize size
        self.width = width
        self.odom = self.read_txt(odom_file)
        self.odom = odom_file
        self.seq_length = seq_length
        self.poses = None
        self.gt_file = gt_file
        if gt_file is not None:
            self.is_gt = True
            self.poses = self.read_gt(gt_file)
        else: self.is_gt = False
        
    def read_txt(self, file):
        odom = []
        with open(file, 'r') as f:
            for line in f:
                values = line.strip().split(',')
                odom.append([float(v) for v in values])

        odom = np.array(odom)
        return odom
    
    def read_gt(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
           
        mats = np.array([np.array(line.split(), dtype=np.float32).reshape(3, 4) for line in lines])
        ts = mats[:, :, 3]
        Rs = mats[:, :, :3]
        yaws = np.arctan2(Rs[:, 1, 0], Rs[:, 0, 0])
        pitches = np.arctan2(-Rs[:, 2, 0], np.sqrt(Rs[:, 2, 1]**2 + Rs[:, 2, 2]**2))
        rolls = np.arctan2(Rs[:, 2, 1], Rs[:, 2, 2])
        poses = np.hstack((ts, rolls.reshape(-1, 1), pitches.reshape(-1, 1), yaws.reshape(-1, 1)))
        return poses
            


    
    def __len__(self):
        return len(self.image_paths)
        
        

    def __getitem__(self, idx):
        
        path = self.image_paths[idx]
        image_names = [os.path.join(path, f) for f in os.listdir(path)]
        sorted_names = sorted(image_names, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        start_id = torch.randint(0, len(sorted_names) - self.seq_length, size=(1,))
        preprocessing = transforms.Compose([
            transforms.CenterCrop((1216,352)),
            # transforms.Resize((304,88)),
            transforms.Resize((608,176)),
            transforms.ToTensor(),
        ])
        tensor_list = []
        
        sorted_list = sorted_names[start_id:self.seq_length + start_id:1]
        for image_name in sorted_list:
            if (len(tensor_list) >= self.seq_length):
                break
            with Image.open(image_name) as img:
                tensor = preprocessing(img)
                tensor_list.append(tensor)

        image_tensor = torch.stack(tensor_list)
        
        odom_path = os.path.join(self.odom, path[-2:]) + ".txt"
        odom = self.read_txt(odom_path)
        gt = None
        if self.is_gt:
            gt_path = os.path.join(self.gt_file, path[-2:]) + ".txt"
            gt = self.read_gt(gt_path)
            gt = gt[start_id:start_id + len(tensor_list), :]
            gt_0 = gt[0,:]
            gt = gt - gt_0
            
        odom = odom[start_id:start_id + len(tensor_list),:]
        odom_0 = odom[0, :]
        odom = odom - odom_0
        return image_tensor, odom, gt
        