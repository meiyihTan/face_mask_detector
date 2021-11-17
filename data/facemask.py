import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from collections import OrderedDict
import xml.etree.ElementTree as ET
import xmltodict

classes = {'background': 0, 'face': 1, 'face_mask': 2}

def is_image(path): 
    if path.split('.')[-1] in ['png', 'jpg', 'jpeg']:
        return True 
    return False

class AnnotationTransform(object): 
    
    def __init__(self, class_to_ind=None): 
        self.class_to_ind = class_to_ind 
    
    def __call__(self, target):
        res = np.empty((0,5))
        d = xmltodict.parse(ET.tostring(target.getroot(), encoding='UTF-8', method='xml')) 
        obj = d['annotation']['object']
        if isinstance(obj, OrderedDict):
            name = obj['name']
            if name == 'face_nask': 
                name = 'face_mask'
            labels = classes[name]
            bbox = obj['bndbox']
            bbox = [int(v) for v in bbox.values()]
#             difficult = [int(obj['difficult'])]
            bbox.append(labels)
            res = np.vstack((res, bbox))
            
        if isinstance(obj, list):
            difficult = [] 
            for i, o in enumerate(obj):
                name = o['name']
                if name == 'face_nask': 
                    name = 'face_mask'
                label = (classes[name])
                bbox = o['bndbox']
                bbox = [int(v) for v in bbox.values()] 
                bbox.append(label)
                difficult.append(int(o['difficult']))
                res = np.vstack((res,bbox))
            
        return res
    
class FaceMaskData(Dataset):
    
    def __init__(self, root, split, preproc=None, target_transform=None): 
        self.split = split.lower() 
        assert self.split in {'train','val'}
        self.root = os.path.join(root, split) 
        self.preproc = preproc
        self.target_transform = target_transform
        self._annopath = [os.path.join(self.root, i) for i in os.listdir(self.root) if is_image(i)==False]
        self._imgpath = [os.path.join(self.root, i) for i in os.listdir(self.root) if is_image(i)]
        
        assert len(self._annopath)==len(self._imgpath)
        
    def __getitem__(self, index):
        target = ET.parse(self._annopath[index])
        
        img = cv2.imread(self._imgpath[index])
        height, width, _ = img.shape
        
        if self.target_transform is not None: 
            target = self.target_transform(target)
            
        if self.preproc is not None: 
            if len(target)==0: 
                pass
            
            else: 
                img, target = self.preproc(img, target)
                
        if self.split == 'val': 
            img = cv2.imread(self._imgpath[index], cv2.IMREAD_COLOR)
            return img, target, self._imgpath[index]
                
        return torch.from_numpy(img), target
    
    def __len__(self):
        return len(self._imgpath)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
