import os 
import numpy as np 
import torch
import cv2
from .nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from data import cfg

# functions for getting predicted bounding boxes and confidence scores for each classes detected

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
classes = {'background': 0, 'face': 1, 'fask_mask': 2}
num_classes = len(classes)

def get_pred(boxes, conf, num_classes, confidence_threshold, nms_threshold):
    """
    select bounding box and prediction that have the highes probabilities 
    Args:
        boxes: (numpy array) decoded point-form boxes
        conf: (Tensor) predicted softmax
        num_classes: (int) number of classes 
        confidence_threshold: (float) threshold to select class
        nms_threshold: (float) threshold to select bounding boxes
    Return: 
        pred: (array) bounding box and prediction that have the highes probabilities 

    """ 
    pred = []
    bbox = boxes.copy() 
    # get prediction scores and bbox for each classes 
    for c in range(1, num_classes):
        pred_scores = conf.clone()
        class_scores = pred_scores[0][:,c].detach().cpu().numpy()
        
        inds = np.where(class_scores > confidence_threshold)[0]        
        class_scores_ = class_scores[inds]
        bbox_ = bbox[inds]

        # keep top-k before NMS 
        order = class_scores_.argsort()[::-1][:5]
        bbox_ = bbox_[order]
        class_scores_ = class_scores_[order]

        dets = np.hstack((bbox_, class_scores_[:, np.newaxis])).astype(np.float32, copy=False) # (b, 5)
        keep = py_cpu_nms(dets, nms_threshold) # use nms to choose the best bbounding box 
        dets = dets[keep]

        for k in range(dets.shape[0]):
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3] 
            ymin += 0.2 * (ymax - ymin + 1)
            score = dets[k, 4]
            
        pred.append(dets)
        
    return pred 

# create ground truth .txt files for model performance evaluation 
def create_gt_file(model, gt_folder, dataset): 
    # gt file - file to store ground-truth .txt files 
    # dataset - val dataset for evaluation 
    for i, (img, label, path) in enumerate(dataset): 
        gt_path = os.path.join(gt_folder, os.path.basename(path).split('.')[0] + '.txt')
        label_ = label[0]
        gt_f = open(gt_path, 'w') 
        # class xmin ymin xmax ymax
        gt_f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(label_[4], label_[0], label_[1], label_[2], label_[3]))
        gt_f.close() 
    print('Done writing ground-truths data into .txt files...')

# create result .txt files for evaluation 
def create_result_file(model, result_folder, dataset, confidence_threshold, nms_threshold): 
    for i, data in enumerate(dataset):
        img_raw, bbox, img_path = data
        bbox_ = bbox[0]
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        loc, conf = model(img) 

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        # convert back to pixel 
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        dets = get_pred(boxes, conf, num_classes, confidence_threshold, nms_threshold)
        # row 1 - class 1 (without mask) row2 2 - class 2 (with mask )

        result_path = os.path.join(result_folder, os.path.basename(img_path).split('.')[0] + '.txt')
        result_f = open(result_path, "w")

        # class , class score, xmin, ymin, xmax, ymax 
        # face - 1, mask - 2
        for j in range(len(dets)): 
            if len(dets[j])!=0: 
                class_score = dets[j][0][-1]
                pred_bbox = dets[j][0][:4]

                pred_bbox = list(map(int, pred_bbox))
                result_f.write('{:.0f} {:.4f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(bbox_[-1], class_score, pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]))

            if len(dets[j])==0: 
                result_f.write('{:.0f} 0 0 0 0 0\n'.format(bbox_[-1]))
        
        result_f.close()
    print('Done writing results data into .txt files...')
    
    

def plot_result(img_raw ,dets): 
    # use cv2 plot outcome with predictions and bounding boxes 
    for i, b in enumerate(dets):
        if len(b)==0: 
            pass 

        if len(b)!=0: 
            b = b[0]
            if i==0: 
                color = (0, 0, 255)
                label = 'no mask'
            elif i==1: 
                color = (255, 0, 0) 
                label = 'with mask'

            text = label + " {:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), color, 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    

    cv2.imshow('res', img_raw)
    cv2.waitKey(0)